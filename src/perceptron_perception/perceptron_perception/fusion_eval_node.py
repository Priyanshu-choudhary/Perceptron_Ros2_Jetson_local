#!/usr/bin/env python3
"""Score object_fusion_node against CARLA ground truth (roadmap Stage 5).

    /perception/objects_json + /perception/object_points
    + /carla/ground_truth/boxes_3d (carla_world) + /carla/ground_truth/lidar_semantic
        -> every report_period, per class and true range: share of boxes that got a lidar
           range, share with no lidar point in the box at all, association correctness,
           range error vs the true closest point of the 3D box, centroid error

Only meaningful with the CARLA reference detector (/carla/ground_truth/boxes_2d), whose
detection ids are actor ids. Definitions: fusion_eval.py.
"""

import json

import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from perceptron_navigation import camera_projection as cp
from perceptron_perception import cloud_eval as ce
from perceptron_perception import fusion_eval as fe
from perceptron_perception import lidar_projection as lp

SEM_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'),
                      ('obj_idx', '<u4'), ('obj_tag', '<u4')])


class FusionEvalNode(Node):

    def __init__(self):
        super().__init__('fusion_eval_node')
        self.declare_parameter('camera_frame', 'camera_optical_link')
        self.declare_parameter('world_frame', 'carla_world')
        self.declare_parameter('report_period', 10.0)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.camera_frame, self.world = p('camera_frame'), p('world_frame')
        self.period, self.total = fe.FusionScores(), fe.FusionScores()
        self.pending = {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(String, '/perception/objects_json', self._json_cb, 10)
        self.create_subscription(PointCloud2, '/perception/object_points', lambda m: self._add('pts', m), qos)
        self.create_subscription(Detection3DArray, '/carla/ground_truth/boxes_3d', lambda m: self._add('gt', m), 10)
        self.create_subscription(PointCloud2, '/carla/ground_truth/lidar_semantic', lambda m: self._add('sem', m), qos)
        self.create_timer(float(p('report_period')), self._report)

    def _json_cb(self, msg):
        data = json.loads(msg.data)
        self._add('json', data, key=tuple(data['stamp']))

    def _add(self, kind, msg, key=None):
        key = key or (msg.header.stamp.sec, msg.header.stamp.nanosec)
        entry = self.pending.setdefault(key, {})
        entry[kind] = msg
        if len(entry) == 4:
            del self.pending[key]
            self._evaluate(key, entry)
        if len(self.pending) > 40:
            for k in sorted(self.pending)[:-40]:
                del self.pending[k]

    def _lookup(self, target, source, key):
        return cp.transform_to_matrix(self.tf_buffer.lookup_transform(
            target, source, Time(seconds=key[0], nanoseconds=key[1]), timeout=Duration(seconds=0.0)))

    def _evaluate(self, key, e):
        data, pts_msg = e['json'], e['pts']
        frame = data['frame']
        try:
            base_from_world = self._lookup(frame, self.world, key)
            world_from_cam = self._lookup(self.world, self.camera_frame, key)
            base_from_lidar = self._lookup(frame, e['sem'].header.frame_id, key)
        except Exception as exc:
            self.get_logger().warn('TF not ready (%s)' % exc, throttle_duration_sec=10.0)
            return
        cam_world = world_from_cam[:3, 3]

        truth = {}
        for d in e['gt'].detections:
            c, q, s = d.bbox.center.position, d.bbox.center.orientation, d.bbox.size
            centre = np.array([c.x, c.y, c.z])
            xy = cp.transform_points(base_from_world, centre[None, :])[0, :2]
            truth[d.id] = {'range': fe.closest_point_distance(cam_world, centre, (q.x, q.y, q.z, q.w),
                                                              (s.x, s.y, s.z)),
                           'xy': xy}

        sem = np.frombuffer(e['sem'].data, dtype=SEM_DTYPE, count=e['sem'].width * e['sem'].height)
        sem_xyz = cp.transform_points(base_from_lidar, np.column_stack([sem['x'], sem['y'], sem['z']]).astype(float))
        f = lp.cloud_fields(pts_msg, ['x', 'y', 'z', 'object'])
        op = np.column_stack([f['x'], f['y'], f['z']]).astype(float)
        o_idx, _, o_ok = ce.match_labels(op, sem_xyz, sem['obj_idx'], sem['obj_tag'])
        index = f['object'].astype(int)

        # "did the lidar hit this actor at all": from the semantic twin, not from points in
        # the box -- the box frustum also holds the wall behind the object
        hit_ids, hit_counts = np.unique(sem['obj_idx'], return_counts=True)
        hits = dict(zip(hit_ids.tolist(), hit_counts.tolist()))
        boxes, objects = [], {}
        for i, o in enumerate(data['objects']):
            try:
                # >= 2: the fusion needs two points before it trusts a range
                lidar_hit = hits.get(int(o['id']), 0) >= 2
            except ValueError:
                lidar_hit = o['points_in_box'] > 0
            boxes.append((o['id'], o['class'], lidar_hit))
            if o['range'] is None:
                continue
            sel = (index == i) & o_ok
            try:
                purity = float((o_idx[sel] == int(o['id'])).mean()) if sel.any() else None
            except ValueError:
                purity = None
            objects[o['id']] = {'range': o['range'], 'xy': (o['x'], o['y']), 'purity': purity}
        for s in (self.period, self.total):
            s.add(boxes, objects, truth)

    def _report(self):
        if not self.period.frames:
            self.get_logger().warn('nothing scored yet: needs carla_bridge (--lidar-gt, --boxes), the Stage 4 '
                                   'preprocessor and object_fusion_node', throttle_duration_sec=30.0)
            return
        self.get_logger().info('\n' + fe.format_summary(self.period.summary(), 'object fusion vs ground truth '
                                                        '(%d frames)' % self.period.frames))
        self.period = fe.FusionScores()

    def destroy_node(self):
        if self.total.frames:
            self.get_logger().info('\n' + fe.format_summary(self.total.summary(), 'FINAL object fusion (%d frames)'
                                                            % self.total.frames))
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FusionEvalNode()
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
