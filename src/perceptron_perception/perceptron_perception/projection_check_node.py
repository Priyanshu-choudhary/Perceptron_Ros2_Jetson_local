#!/usr/bin/env python3
"""Measure lidar-to-camera alignment in pixels, per distance (Task 01 pass test).

    /points + /depth (same pose as the RGB camera) + CameraInfo
        -> project exactly as lidar_camera_overlay_node does
        -> AlignmentAccumulator -> error vector (du, dv) per range bin, every report_period

The method, and why it is one shift per bin rather than a per-point error, is
in lidar_projection.py. In CARLA the target is |err| under 1 px in every bin:
the simulator has no calibration error, so anything larger is a bug in the
frame chain, the intrinsics or the timing, and it would be in the robot too.

To prove the check can see an error, run it once with perturb_rpy_deg set to
[0, 1, 0]. That is about optical +Y, i.e. a 1 deg camera yaw (optical +Z would
roll the image instead), and should come out near fx * tan(1 deg) ~= 11 px at
fx = 640 (CARLA 1280 px, 90 deg FOV).
"""

import message_filters
import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from perceptron_navigation import camera_projection as cp
from perceptron_perception import lidar_projection as lp


class ProjectionCheckNode(Node):

    def __init__(self):
        super().__init__('projection_check_node')

        self.declare_parameter('depth_topic', '/carla/camera/depth')
        self.declare_parameter('cloud_topic', '/points')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('camera_optical_frame', 'camera_optical_link')
        self.declare_parameter('min_z', 0.15)
        self.declare_parameter('max_radius', 2.35)
        self.declare_parameter('max_range', 100.0)
        self.declare_parameter('max_shift', 15)
        self.declare_parameter('rel_tol', 0.01)
        self.declare_parameter('abs_tol', 0.05)
        self.declare_parameter('range_bins', list(lp.DEFAULT_RANGE_BINS))
        self.declare_parameter('report_period', 5.0)
        # (2*max_shift+1)^2 depth lookups per point per frame; every 4th frame is plenty.
        self.declare_parameter('check_every', 4)
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('perturb_rpy_deg', [0.0, 0.0, 0.0])
        self.declare_parameter('perturb_xyz', [0.0, 0.0, 0.0])

        p = self.get_parameter
        self.optical_frame = p('camera_optical_frame').value
        self.min_z = float(p('min_z').value)
        self.max_radius = float(p('max_radius').value)
        self.max_range = float(p('max_range').value)
        self.check_every = max(1, int(p('check_every').value))
        acc_args = dict(bins=tuple(float(b) for b in p('range_bins').value),
                        max_shift=int(p('max_shift').value),
                        rel_tol=float(p('rel_tol').value), abs_tol=float(p('abs_tol').value))
        self.period = lp.AlignmentAccumulator(**acc_args)
        self.total = lp.AlignmentAccumulator(**acc_args)
        self.pairs = 0
        self.perturb_rpy = list(p('perturb_rpy_deg').value)
        self.perturb = lp.perturbation(self.perturb_rpy, p('perturb_xyz').value)

        self.camera_info = None
        self.create_subscription(CameraInfo, p('camera_info_topic').value, self._info_cb, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        d_sub = message_filters.Subscriber(self, Image, p('depth_topic').value,
                                           qos_profile=sensor_qos)
        pc_sub = message_filters.Subscriber(self, PointCloud2, p('cloud_topic').value,
                                            qos_profile=sensor_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [d_sub, pc_sub], queue_size=10, slop=float(p('sync_slop').value))
        self.sync.registerCallback(self._pair_cb)

        self.create_timer(float(p('report_period').value), self._report)

    def _info_cb(self, msg):
        self.camera_info = msg

    def _pair_cb(self, depth_msg, cloud_msg):
        self.pairs += 1
        if self.pairs % self.check_every:
            return
        if self.camera_info is None:
            self.get_logger().warn('waiting for CameraInfo', throttle_duration_sec=5.0)
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.optical_frame, cloud_msg.header.frame_id, rclpy.time.Time())
        except Exception as exc:
            self.get_logger().warn('No transform %s -> %s (%s)'
                                   % (cloud_msg.header.frame_id, self.optical_frame, exc),
                                   throttle_duration_sec=10.0)
            return
        K, D, _ = lp.intrinsics_from_camera_info(self.camera_info)
        depth = lp.depth_to_array(depth_msg)
        h, w = depth.shape

        pts_opt = cp.transform_points(self.perturb @ cp.transform_to_matrix(tf),
                                      lp.cloud_to_xyz(cloud_msg))
        pts_opt = pts_opt[cp.visible_mask(pts_opt, self.min_z, self.max_radius, self.max_range)]
        uv = cp.project_points(pts_opt, K, D)
        on = cp.inside_image(uv, w, h)
        uv, pts_opt = uv[on], pts_opt[on]

        ranges = np.linalg.norm(pts_opt, axis=1)
        self.period.add(uv, pts_opt[:, 2], ranges, depth)

    def _table(self, acc, title):
        if acc.frames == 0:
            return None
        return lp.format_alignment(acc.results(), '%s (%d frames, perturb_rpy_deg=%s)'
                                   % (title, acc.frames, self.perturb_rpy))

    def _report(self):
        table = self._table(self.period, 'lidar->camera alignment')
        if table is None:
            if self.pairs:
                self.get_logger().warn('No depth+cloud pairs checked yet -- waiting for '
                                       'CameraInfo / TF?', throttle_duration_sec=10.0)
            return
        self.get_logger().info('\n' + table)
        self.total.merge(self.period)
        self.period.reset()

    def destroy_node(self):
        self.total.merge(self.period)
        self.period.reset()
        table = self._table(self.total, 'FINAL lidar->camera alignment')
        if table:
            self.get_logger().info('\n' + table)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ProjectionCheckNode()
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
