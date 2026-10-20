#!/usr/bin/env python3
"""Score the EKF outputs against CARLA's perfect pose (roadmap Stage 3).

    /carla/ground_truth/ego_odom  (carla_world)   -- evaluation only, never an EKF input
    /odometry/local   (odom)  --TF carla_world<-odom-->  error vs ground truth
    /odometry/global  (map)   --TF carla_world<-map--->  error vs ground truth

Each estimate is moved into carla_world through TF (carla_world -> map is the
bridge's static anchor at the start pose), so a wrong yaw convention or frame
shows up here as a large error. /odometry/local is scored as dead reckoning
from the start, /odometry/global as the GPS-corrected pose. Ground truth is
interpolated to each estimate's stamp.

Every report_period it logs one line per topic (position RMSE / max / final,
drift as % of distance travelled, heading RMSE / max, update rate) and on
shutdown the totals, also written to ~/carla_logs/loc_eval_<time>.csv.
"""

import csv
import os
import time

import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from perceptron_carla import loc_metrics as lm
from perceptron_navigation import camera_projection as cp


class LocalizationEvalNode(Node):

    def __init__(self):
        super().__init__('localization_eval_node')
        self.declare_parameter('ground_truth_topic', '/carla/ground_truth/ego_odom')
        self.declare_parameter('estimate_topics', ['/odometry/local', '/odometry/global'])
        self.declare_parameter('world_frame', 'carla_world')
        self.declare_parameter('report_period', 10.0)
        self.declare_parameter('log_dir', os.path.expanduser('~/carla_logs'))

        p = self.get_parameter
        self.world_frame = p('world_frame').value
        self.map_frame, self.odom_frame = 'map', 'odom'
        self.gt = lm.GroundTruthBuffer()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, p('ground_truth_topic').value, self._gt_cb, 50)

        self.period, self.total = {}, {}
        for topic in p('estimate_topics').value:
            self.period[topic] = lm.ErrorStats()
            self.total[topic] = lm.ErrorStats()
            self.create_subscription(Odometry, topic, lambda m, t=topic: self._est_cb(t, m), 50)

        os.makedirs(p('log_dir').value, exist_ok=True)
        self.csv_path = os.path.join(p('log_dir').value, time.strftime('loc_eval_%Y%m%d_%H%M%S.csv'))
        self.create_timer(float(p('report_period').value), self._report)

    @staticmethod
    def _t(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _gt_cb(self, msg):
        q = msg.pose.pose.orientation
        self.gt.add(self._t(msg.header.stamp), msg.pose.pose.position.x, msg.pose.pose.position.y,
                    lm.yaw_from_quat(q.x, q.y, q.z, q.w))

    def _est_cb(self, topic, msg):
        t = self._t(msg.header.stamp)
        gt = self.gt.at(t)
        if gt is None:
            return
        # odom is scored against where it STARTED, which is where map starts too (both EKFs
        # begin at the origin with the IMU's ENU yaw). Going through the live map -> odom
        # would let the global EKF's GPS corrections leak into the local EKF's score.
        source = self.map_frame if msg.header.frame_id == self.odom_frame else msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, source, Time())
        except Exception as exc:
            self.get_logger().warn('no TF %s -> %s (%s)' % (msg.header.frame_id, self.world_frame, exc),
                                   throttle_duration_sec=10.0)
            return
        M = cp.transform_to_matrix(tf)
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        x, y, _ = cp.transform_points(M, np.array([[pos.x, pos.y, pos.z]]))[0]
        yaw = lm.wrap_angle(lm.yaw_from_quat(q.x, q.y, q.z, q.w)
                            + np.arctan2(M[1, 0], M[0, 0]))
        for store in (self.period, self.total):
            store[topic].add(t, (x, y, yaw), gt)

    def _report(self):
        lines = [lm.format_summary(topic, s.summary()) for topic, s in self.period.items()]
        totals = self._write_totals()
        self.get_logger().info('localization error, last period:\n  ' + '\n  '.join(lines)
                               + '\nsince start (%s):\n  ' % self.csv_path + '\n  '.join(totals))
        for topic in self.period:
            self.period[topic] = lm.ErrorStats()

    def _write_totals(self):
        """Rewrite the CSV with the totals so far; a killed run still leaves its result."""
        lines = []
        tmp = self.csv_path + '.tmp'
        with open(tmp, 'w', newline='') as f:
            w = csv.writer(f)
            header = None
            for topic, s in self.total.items():
                summ = s.summary()
                lines.append(lm.format_summary(topic, summ))
                if summ:
                    if header is None:
                        header = ['topic'] + list(summ)
                        w.writerow(header)
                    w.writerow([topic] + [summ[k] for k in header[1:]])
        os.replace(tmp, self.csv_path)       # atomic: a Ctrl-C mid-write keeps the last good file
        return lines

    def destroy_node(self):
        self._write_totals()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationEvalNode()
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
