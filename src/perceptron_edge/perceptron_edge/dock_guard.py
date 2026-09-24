#!/usr/bin/env python3
"""dock_guard - lets Nav2 start a goal while the robot sits on the charger.

On the charger the robot's footprint already touches the dock, so every Nav2
motion refuses to start: the rotation shim, DWB, Spin and BackUp all check the
footprint against the costmap first ("Collision Ahead - Exiting
DriveOnHeading", measured). The planner still finds a path, so a goal is
accepted, cycles through its recoveries without moving and fails a minute later.

The navigate-to-pose behaviour tree (behavior_trees/navigate_to_pose.xml) calls
/dock/undock_if_docked before anything else. This node answers it:

    not on the charger  -> returns at once; the goal starts with no delay
    on the charger      -> reverses straight off first (the `dock undock`
                           routine: 0.35 m, lidar watching behind, no costmap
                           check), then returns and Nav2 plans from free space

"On the charger" is /battery/state CHARGING, or AMCL within pose_tolerance of
the pose recorded by `robot dock-teach` (a full pack draws no charge current).

It must be up before bt_navigator activates: a behaviour tree service node
whose server is missing fails the whole tree at load time, so edge.launch.py
starts this first and nav2_edge.yaml gives bt_navigator 5 s to find it.
"""

import math
import os
import subprocess
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState
from std_srvs.srv import Empty


class DockGuard(Node):

    def __init__(self):
        super().__init__('dock_guard')
        self.declare_parameter('pose_tolerance', 0.12)      # m from the taught dock pose
        self.declare_parameter('yaw_tolerance', 0.35)       # rad
        self.declare_parameter('undock_distance', 0.35)     # m
        self.declare_parameter('station_file', os.path.join(
            os.environ.get('PERCEPTRON_WS', '/opt/perceptron_ws'), 'config', 'dock_station.yaml'))
        self.battery = None      # (wall time, power_supply_status)
        self.pose = None         # (x, y, yaw) from AMCL
        self.tool = os.path.join(get_package_prefix('perceptron_edge'), 'lib', 'perceptron_edge', 'dock')
        self.create_subscription(BatteryState, '/battery/state', self._battery_cb, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._pose_cb, latched)
        self.create_service(Empty, '/dock/undock_if_docked', self._on_request)

    def _battery_cb(self, msg):
        self.battery = (time.time(), msg.power_supply_status)

    def _pose_cb(self, msg):
        q = msg.pose.pose.orientation
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                     math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

    def _on_charger(self):
        """Why the robot counts as docked, or None."""
        if (self.battery is not None and time.time() - self.battery[0] < 5.0
                and self.battery[1] == BatteryState.POWER_SUPPLY_STATUS_CHARGING):
            return 'charging'
        if self.pose is None:
            return None
        try:
            with open(self.get_parameter('station_file').value) as f:
                dock = (yaml.safe_load(f) or {}).get('dock_map_pose')
        except (OSError, yaml.YAMLError):
            return None
        if not dock:
            return None
        dist = math.hypot(self.pose[0] - float(dock['x']), self.pose[1] - float(dock['y']))
        dyaw = abs(math.atan2(math.sin(self.pose[2] - float(dock['yaw'])),
                              math.cos(self.pose[2] - float(dock['yaw']))))
        if (dist < self.get_parameter('pose_tolerance').value
                and dyaw < self.get_parameter('yaw_tolerance').value):
            return 'at the dock pose (%.2f m from it)' % dist
        return None

    def _on_request(self, request, response):
        why = self._on_charger()
        if why is None:
            return response
        distance = float(self.get_parameter('undock_distance').value)
        self.get_logger().info('goal while on the charger (%s): backing off %.2f m first'
                               % (why, distance))
        try:
            done = subprocess.run([self.tool, 'undock', str(distance)], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
            log = self.get_logger().info if done.returncode == 0 else self.get_logger().warn
            for line in done.stdout.strip().splitlines()[-3:]:
                log('undock: ' + line)
        except subprocess.TimeoutExpired:
            self.get_logger().warn('undock did not finish in 30 s')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DockGuard()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
