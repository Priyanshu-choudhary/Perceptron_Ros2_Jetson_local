#!/usr/bin/env python3
"""Yield to predicted motion: a Nav2 speed limit from Stage 7 predictions (Stage 11).

    /perception/predicted_objects (PredictedObjectArray, odom)  +  /plan (Nav2 path)  +  TF
      -> /speed_limit       nav2_msgs/SpeedLimit (absolute m/s; 0 = no limit) for controller_server
      -> /yield/status      String JSON: limit, binding conflict (time, distance along the path)

Every predicted position of a moving object that falls in the robot's corridor (the plan ahead,
widened by half the robot width + the object's radius + a margin) caps the speed so the robot
neither reaches that spot before the object has crossed nor loses the ability to stop short of
it; see yielding.py. The cap drops at once and rises slowly. Nav2's collision monitor stays the
last line of defence for what no prediction foresaw.
"""

import json
import math

import numpy as np
import rclpy
import tf2_ros
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String

from perceptron_msgs.msg import PredictedObject, PredictedObjectArray
from perceptron_navigation import camera_projection as cp
from perceptron_navigation import yielding


class PredictiveSpeedLimitNode(Node):

    def __init__(self):
        super().__init__('predictive_speed_limit_node')
        d = self.declare_parameter
        d('frame', 'odom')
        d('robot_frame', 'base_footprint')
        d('half_width', 0.8)            # m, robot
        d('nose', 2.1)                  # m, front of the robot ahead of base_footprint
        d('gap', 1.5)                   # m kept between the nose and a crossing object
        d('margin', 0.5)
        d('decel', 2.0)                 # m/s^2, comfortable braking
        d('rise', 1.0)                  # m/s per s the limit may come back up
        d('release_speed', 4.0)         # a limit at or above this is no limit
        d('corridor_length', 30.0)
        d('rate', 10.0)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.frame, self.robot_frame = p('frame'), p('robot_frame')
        self.kw = dict(half_width=float(p('half_width')), nose=float(p('nose')), gap=float(p('gap')),
                       margin=float(p('margin')), decel=float(p('decel')))
        self.length = float(p('corridor_length'))
        self.dt = 1.0 / float(p('rate'))
        self.limiter = yielding.Limiter(rise=float(p('rise')), release=float(p('release_speed')))
        self.predictions = None
        self.plan = None
        self.yielding = False
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(PredictedObjectArray, '/perception/predicted_objects',
                                 lambda m: setattr(self, 'predictions', m), 10)
        self.create_subscription(Path, '/plan', lambda m: setattr(self, 'plan', m), 10)
        self.pub = self.create_publisher(SpeedLimit, '/speed_limit', 10)
        self.pub_status = self.create_publisher(String, '/yield/status', 10)
        self.create_timer(self.dt, self._tick)

    def _corridor(self, robot):
        x, y, yaw = robot
        if self.plan is not None and len(self.plan.poses) >= 2:
            try:
                M = cp.transform_to_matrix(self.tf_buffer.lookup_transform(
                    self.frame, self.plan.header.frame_id, Time()))
                pts = np.array([[q.pose.position.x, q.pose.position.y, 0.0] for q in self.plan.poses])
                pts = cp.transform_points(M, pts)[:, :2]
                k = int(np.argmin(np.linalg.norm(pts - (x, y), axis=1)))
                ahead = pts[k:]
                if len(ahead) >= 2:
                    return yielding.Corridor(np.vstack([[x, y], ahead]), self.length)
            except Exception:
                pass
        return yielding.Corridor.straight(x, y, yaw, self.length)

    def _objects(self):
        if self.predictions is None:
            return []
        return [{'path': [(q.x, q.y) for q in o.path], 'path_dt': o.path_dt, 'radius': o.radius}
                for o in self.predictions.objects if o.model != PredictedObject.STATIC and o.path]

    def _tick(self):
        try:
            M = cp.transform_to_matrix(self.tf_buffer.lookup_transform(self.frame, self.robot_frame, Time()))
        except Exception:
            return
        robot = (M[0, 3], M[1, 3], math.atan2(M[1, 0], M[0, 0]))
        cap, why = yielding.speed_cap(self._corridor(robot), self._objects(), **self.kw)
        v = self.limiter.update(cap, self.dt)
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = False
        msg.speed_limit = 0.0 if v is None else max(v, 0.01)   # 0 means "no limit" to Nav2
        self.pub.publish(msg)
        if (v is not None) != self.yielding:
            self.yielding = v is not None
            if self.yielding and why:
                self.get_logger().info('yielding: %.2f m/s, a moving object will be %.1f m along the path in %.1f s'
                                       % (v, why[1], why[0]))
            else:
                self.get_logger().info('path clear: no speed limit')
        self.pub_status.publish(String(data=json.dumps({
            'limit': None if v is None else round(v, 2), 'cap': None if cap is None else round(cap, 2),
            'conflict': None if why is None else [round(w, 2) for w in why]})))


def main(args=None):
    rclpy.init(args=args)
    node = PredictiveSpeedLimitNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
