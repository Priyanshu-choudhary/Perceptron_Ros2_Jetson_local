#!/usr/bin/env python3
"""
GPS Waypoint Client for Perceptron Rover.
Sends single GPS goals or routes to Nav2.

Usage:
    ros2 run perceptron_edge gps_waypoint waypoint <LAT> <LON> [YAW_DEG]
    ros2 run perceptron_edge gps_waypoint route <ROUTE.YAML>
"""

import math
import sys
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import yaml
from geographic_msgs.msg import GeoPoint
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
from robot_localization.srv import FromLL


def euler_to_quaternion(yaw_rad: float):
    return (0.0, 0.0, math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5))


class GPSWaypointClient(Node):
    def __init__(self):
        super().__init__('gps_waypoint_client')
        self.from_ll_client = self.create_client(FromLL, '/fromLL')
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.nav_through_poses_client = ActionClient(self, NavigateThroughPoses, '/navigate_through_poses')

    def wait_for_services(self, timeout_sec=10.0) -> bool:
        self.get_logger().info("Waiting for /fromLL and Nav2 action servers...")
        if not self.from_ll_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("/fromLL service (navsat_transform_node) not available!")
            return False
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error("Nav2 /navigate_to_pose action server not available!")
            return False
        return True

    def convert_lat_lon_to_map(self, lat: float, lon: float, alt: float = 0.0) -> Tuple[float, float]:
        req = FromLL.Request()
        req.ll_point = GeoPoint(latitude=lat, longitude=lon, altitude=alt)
        future = self.from_ll_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None:
            raise RuntimeError(f"Failed to convert ({lat}, {lon}) to map coordinates via /fromLL")
        return res.map_point.x, res.map_point.y

    def send_single_waypoint(self, lat: float, lon: float, yaw_deg: float = 0.0) -> bool:
        if not self.wait_for_services():
            return False

        map_x, map_y = self.convert_lat_lon_to_map(lat, lon)
        self.get_logger().info(f"Target GPS: Lat={lat:.7f}, Lon={lon:.7f} -> Map Cartesian: X={map_x:.2f}m, Y={map_y:.2f}m")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = map_x
        goal_msg.pose.pose.position.y = map_y
        goal_msg.pose.pose.position.z = 0.0

        yaw_rad = math.radians(yaw_deg)
        qx, qy, qz, qw = euler_to_quaternion(yaw_rad)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info("Sending goal to Nav2...")
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected the goal!")
            return False

        self.get_logger().info("Goal accepted by Nav2. Moving to waypoint...")
        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        result = res_future.result()

        if result.status == 4:  # STATUS_SUCCEEDED
            self.get_logger().info("Waypoint reached successfully!")
            return True
        else:
            self.get_logger().warn(f"Goal completed with status code: {result.status}")
            return False

    def send_route(self, route_yaml_path: str) -> bool:
        if not self.wait_for_services():
            return False

        with open(route_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        waypoints = data.get('waypoints', [])
        if not waypoints:
            self.get_logger().error(f"No waypoints found in {route_yaml_path}")
            return False

        poses: List[PoseStamped] = []
        for i, wp in enumerate(waypoints):
            lat = float(wp['lat'])
            lon = float(wp['lon'])
            yaw_deg = float(wp.get('yaw', 0.0))
            map_x, map_y = self.convert_lat_lon_to_map(lat, lon)
            self.get_logger().info(f"WP {i+1}: ({lat:.7f}, {lon:.7f}) -> ({map_x:.2f}, {map_y:.2f})")

            p = PoseStamped()
            p.header.frame_id = 'map'
            p.header.stamp = self.get_clock().now().to_msg()
            p.pose.position.x = map_x
            p.pose.position.y = map_y
            qx, qy, qz, qw = euler_to_quaternion(math.radians(yaw_deg))
            p.pose.orientation.x = qx
            p.pose.orientation.y = qy
            p.pose.orientation.z = qz
            p.pose.orientation.w = qw
            poses.append(p)

        goal_msg = NavigateThroughPoses.Goal()
        goal_msg.poses = poses

        self.get_logger().info(f"Sending route with {len(poses)} waypoints to Nav2...")
        send_goal_future = self.nav_through_poses_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected the route!")
            return False

        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        result = res_future.result()
        return result.status == 4

    def _feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        rem = getattr(fb, 'distance_remaining', None)
        if rem is not None:
            sys.stdout.write(f"\r[Nav2 Progress] Distance to waypoint: {rem:.2f} m   ")
            sys.stdout.flush()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  robot waypoint <LAT> <LON> [YAW_DEG]")
        print("  robot route <ROUTE.YAML>")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    rclpy.init()
    node = GPSWaypointClient()

    try:
        if cmd == 'waypoint':
            if len(sys.argv) < 4:
                print("Usage: waypoint <LAT> <LON> [YAW_DEG]")
                sys.exit(1)
            lat = float(sys.argv[2])
            lon = float(sys.argv[3])
            yaw = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
            success = node.send_single_waypoint(lat, lon, yaw)
            sys.exit(0 if success else 1)
        elif cmd == 'route':
            if len(sys.argv) < 3:
                print("Usage: route <ROUTE.YAML>")
                sys.exit(1)
            path = sys.argv[2]
            success = node.send_route(path)
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
