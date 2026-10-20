#!/usr/bin/env python3
"""Score lidar_preprocessor_node against CARLA ground truth (roadmap Stage 4).

    /carla/ground_truth/lidar_semantic  + /perception/points_ground
    + /perception/points_obstacles (cluster field) + /carla/bridge_info (ego id)
    + /perception/lidar_preprocess_stats
        -> every report_period: ground precision/recall, host-vehicle points removed,
           actor recall / over-split / under-split per class and range, processing
           time, and the fitted ground plane (task 4.1 extrinsic check)

Method and definitions: cloud_eval.py. Evaluation only; never part of the
robot pipeline.
"""

import json

import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String

from perceptron_navigation import camera_projection as cp
from perceptron_perception import cloud_eval as ce

SEM_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'),
                      ('obj_idx', '<u4'), ('obj_tag', '<u4')])


def cloud_fields(msg, names):
    """Columns by name from a PointCloud2 of float32 fields."""
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({'names': names, 'formats': ['<f4'] * len(names),
                   'offsets': [off[n] for n in names], 'itemsize': msg.point_step})
    return np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)


class CloudEvalNode(Node):

    def __init__(self):
        super().__init__('cloud_eval_node')
        self.declare_parameter('gt_topic', '/carla/ground_truth/lidar_semantic')
        self.declare_parameter('match_tolerance', 0.1)
        self.declare_parameter('min_hits', 3)
        self.declare_parameter('report_period', 10.0)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.tol = float(p('match_tolerance'))
        self.min_hits = int(p('min_hits'))
        self.period = ce.FrameScores(self.min_hits)
        self.total = ce.FrameScores(self.min_hits)
        self.stats = []
        self.ego_id = None
        self.pending = {}

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PointCloud2, p('gt_topic'), lambda m: self._add('gt', m), qos)
        self.create_subscription(PointCloud2, '/perception/points_ground', lambda m: self._add('ground', m), qos)
        self.create_subscription(PointCloud2, '/perception/points_obstacles', lambda m: self._add('obst', m), qos)
        self.create_subscription(Float32MultiArray, '/perception/lidar_preprocess_stats',
                                 lambda m: self.stats.append(list(m.data)), 10)
        self.create_subscription(String, '/carla/bridge_info', self._info_cb, latched)
        self.create_timer(float(p('report_period')), self._report)

    def _info_cb(self, msg):
        try:
            self.ego_id = int(json.loads(msg.data)['ego_id'])
        except (ValueError, KeyError):
            pass

    def _add(self, kind, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        entry = self.pending.setdefault(key, {})
        entry[kind] = msg
        if len(entry) == 3:
            del self.pending[key]
            self._evaluate(entry)
        if len(self.pending) > 30:                    # drop frames that never completed
            for k in sorted(self.pending)[:-30]:
                del self.pending[k]

    def _evaluate(self, e):
        gt = np.frombuffer(e['gt'].data, dtype=SEM_DTYPE, count=e['gt'].width * e['gt'].height)
        frame = e['obst'].header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(frame, e['gt'].header.frame_id, rclpy.time.Time())
        except Exception as exc:
            self.get_logger().warn('no TF for ground truth (%s)' % exc, throttle_duration_sec=10.0)
            return
        gxyz = cp.transform_points(cp.transform_to_matrix(tf),
                                   np.column_stack([gt['x'], gt['y'], gt['z']]).astype(np.float64))
        g = cloud_fields(e['ground'], ['x', 'y', 'z'])
        o = cloud_fields(e['obst'], ['x', 'y', 'z', 'cluster'])
        gp = np.column_stack([g['x'], g['y'], g['z']]).astype(np.float64)
        op = np.column_stack([o['x'], o['y'], o['z']]).astype(np.float64)
        args = (gxyz, gt['obj_idx'], gt['obj_tag'], self.tol)
        g_lab = ce.match_labels(gp, *args)
        o_lab = ce.match_labels(op, *args)
        cl = o['cluster'].astype(np.int64)
        for s in (self.period, self.total):
            s.add(g_lab, o_lab, cl, op[:, :2], gt['obj_idx'], self.ego_id)

    def _stats_lines(self):
        if not self.stats:
            return ['  (no /perception/lidar_preprocess_stats)']
        a = np.array(self.stats)
        return ['  timing   processing mean %.1f ms, p95 %.1f ms, max %.1f ms over %d clouds'
                % (a[:, 0].mean(), np.percentile(a[:, 0], 95), a[:, 0].max(), len(a)),
                '  points   in %.0f, kept %.0f, ground %.0f, obstacles %.0f, clusters %.1f (means per cloud)'
                % tuple(a[:, 1:6].mean(axis=0)),
                '  plane    tilt median %.2f deg, height at base_footprint median %+.3f m (task 4.1: ~0 if the '
                'lidar extrinsic is right)' % (np.nanmedian(a[:, 6]), np.nanmedian(a[:, 7]))]

    def _report(self):
        if self.period.frames == 0:
            self.get_logger().warn('nothing evaluated yet: need the bridge (with --lidar-gt) and '
                                   'lidar_preprocessor_node', throttle_duration_sec=30.0)
            return
        text = ce.format_summary(self.period.summary(), 'lidar preprocessing vs ground truth (%d frames)'
                                 % self.period.frames)
        self.get_logger().info('\n' + text + '\n' + '\n'.join(self._stats_lines()))
        self.period = ce.FrameScores(self.min_hits)
        self.stats = []

    def destroy_node(self):
        if self.total.frames:
            self.get_logger().info('\n' + ce.format_summary(
                self.total.summary(), 'FINAL lidar preprocessing (%d frames)' % self.total.frames))
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CloudEvalNode()
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
