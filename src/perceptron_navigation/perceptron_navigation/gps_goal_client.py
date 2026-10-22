#!/usr/bin/env python3
"""Send a GPS goal to gps_goal_node from the command line.

    ros2 run perceptron_navigation gps_goal 19.1351 72.9122        # latitude longitude
    ros2 run perceptron_navigation gps_goal --ahead 40             # 40 m ahead of the robot
    ros2 run perceptron_navigation gps_goal --ahead 30 --left 5
    ros2 run perceptron_navigation gps_goal --cancel
    ros2 run perceptron_navigation gps_goal --watch                # print /gps_goal/status

--ahead/--left are in base_footprint at the moment the node receives them; the node logs the
GPS coordinate it turned them into, so the same goal can be given again by latitude/longitude.
"""

import argparse
import json
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Empty, String


def main(argv=None):
    p = argparse.ArgumentParser(description='send a GPS goal to gps_goal_node')
    p.add_argument('lat', nargs='?', type=float)
    p.add_argument('lon', nargs='?', type=float)
    p.add_argument('--ahead', type=float, help='metres ahead of the robot instead of lat/lon')
    p.add_argument('--left', type=float, default=0.0, help='with --ahead: metres to the left')
    p.add_argument('--cancel', action='store_true')
    p.add_argument('--watch', action='store_true', help='print the goal status until it ends')
    args, _ = p.parse_known_args(argv)
    if not args.cancel and args.ahead is None and (args.lat is None or args.lon is None) and not args.watch:
        p.error('give LAT LON, --ahead D, --cancel or --watch')

    rclpy.init()
    node = Node('gps_goal_client')
    if args.cancel:
        pub, msg = node.create_publisher(Empty, '/gps_goal/cancel', 10), Empty()
    elif args.ahead is not None:
        pub, msg = node.create_publisher(PointStamped, '/gps_goal/relative', 10), PointStamped()
        msg.header.frame_id = 'base_footprint'
        msg.point.x, msg.point.y = args.ahead, args.left
    elif args.lat is not None:
        pub, msg = node.create_publisher(NavSatFix, '/gps_goal', 10), NavSatFix()
        msg.latitude, msg.longitude = args.lat, args.lon
    else:
        pub = None
    if pub is not None:
        end = time.time() + 5.0
        while pub.get_subscription_count() == 0 and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            print('gps_goal_node is not running')
            return 1
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.3)
        print('sent')
    if args.watch or (pub is not None and not args.cancel):
        last = {}

        def cb(m):
            last['s'] = json.loads(m.data)
        node.create_subscription(String, '/gps_goal/status', cb, 10)
        seen_driving = False
        try:
            while True:
                rclpy.spin_once(node, timeout_sec=1.0)
                s = last.get('s')
                if not s:
                    continue
                print('  %-9s %s m left' % (s['state'], s['distance_left']))
                seen_driving |= s['state'] == 'driving'
                if seen_driving and s['state'] != 'driving':
                    break
        except KeyboardInterrupt:
            pass
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
