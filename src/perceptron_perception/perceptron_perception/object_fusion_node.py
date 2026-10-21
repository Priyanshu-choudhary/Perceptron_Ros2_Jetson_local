#!/usr/bin/env python3
"""Camera detections + lidar clusters -> objects with class, 3D position and range (Stage 5).

    detections_topic  vision_msgs/Detection2DArray (any 2D detector; in CARLA the reference
                      boxes /carla/ground_truth/boxes_2d until the own model is ready)
  + /perception/points_obstacles  (Stage 4: x, y, z, cluster in base_footprint)
  + CameraInfo (or the measured calibration parameters) + TF
      -> /perception/objects        vision_msgs/Detection3DArray in base_footprint: one per box
                                    that got lidar points; id and class from the 2D detection,
                                    bbox = extent of its points, hypothesis pose = centroid with
                                    covariance
      -> /perception/objects_json   std_msgs/String, every box: id, class, score, range (m to the
                                    nearest surface from the camera, null without lidar), x/y/z,
                                    points, how it was associated, lidar points inside the box
      -> /perception/object_points  PointCloud2 x, y, z, object (index into objects_json)
      -> /perception/object_markers MarkerArray
      -> /perception/objects_image  camera image with boxes, class and range (when subscribed)

Association and the range definition: object_fusion.py. Detections and cloud are paired by
stamp; the camera->base_footprint transform is static, so the latest one is used.
"""

import array
import json
import time

import cv2
import message_filters
import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from perceptron_navigation import camera_projection as cp
from perceptron_navigation import marker_map as mm
from perceptron_perception import lidar_projection as lp
from perceptron_perception import object_fusion as of

COLOURS = {'pedestrian': (60, 60, 255), 'traffic_cone': (0, 200, 255), 'box': (0, 200, 255)}


def box_colour(cls):
    return COLOURS.get(cls, (80, 220, 80) if cls in ('car', 'truck', 'van', 'bus', 'motorcycle',
                                                      'bicycle', 'vehicle') else (0, 200, 255))


class ObjectFusionNode(Node):

    def __init__(self):
        super().__init__('object_fusion_node')
        d = self.declare_parameter
        d('detections_topic', '/perception/detections_2d')
        d('obstacles_topic', '/perception/points_obstacles')
        d('camera_info_topic', '')
        d('image_topic', '/camera/image_raw')
        d('camera_optical_frame', 'camera_optical_link')
        d('camera_matrix', [410.9212916385953, 0.0, 634.8042096493966,
                            0.0, 411.4395116912116, 347.58359871233904, 0.0, 0.0, 1.0])
        d('dist_coeffs', [0.030287865182229395, -0.002860707267771121, -0.004223044189411946,
                          -0.0003374289867825264, -0.0003820853552867496])
        d('calibration_width', 1280)
        d('calibration_height', 720)
        d('min_affinity', 0.2)
        d('box_shrink', 0.05)
        d('join_depth', 2.5)
        d('fallback_min_points', 2)
        d('size_gate', True)                   # refuse points farther than a pedestrian box allows
        d('size_gate_factor', 1.15)
        d('min_score', 0.0)
        d('sync_slop', 0.05)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.optical = p('camera_optical_frame')
        self.base_K = np.array(p('camera_matrix'), dtype=np.float64).reshape(3, 3)
        self.base_D = np.array(p('dist_coeffs'), dtype=np.float64)
        self.calib = (int(p('calibration_width')), int(p('calibration_height')))
        self.assoc = dict(min_affinity=float(p('min_affinity')), shrink=float(p('box_shrink')),
                          join_depth=float(p('join_depth')), fallback_min_points=int(p('fallback_min_points')))
        self.min_score = float(p('min_score'))
        self.size_gate = bool(p('size_gate'))
        self.size_gate_factor = float(p('size_gate_factor'))

        self.camera_info = None
        if p('camera_info_topic'):
            self.create_subscription(CameraInfo, p('camera_info_topic'), self._info_cb, 10)
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.images = {}
        self.create_subscription(Image, p('image_topic'), self._image_cb, sensor_qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub_obj = self.create_publisher(Detection3DArray, '/perception/objects', 10)
        self.pub_json = self.create_publisher(String, '/perception/objects_json', 10)
        self.pub_pts = self.create_publisher(PointCloud2, '/perception/object_points', sensor_qos)
        self.pub_mk = self.create_publisher(MarkerArray, '/perception/object_markers', 10)
        self.pub_img = self.create_publisher(Image, '/perception/objects_image', sensor_qos)

        det_sub = message_filters.Subscriber(self, Detection2DArray, p('detections_topic'))
        pc_sub = message_filters.Subscriber(self, PointCloud2, p('obstacles_topic'), qos_profile=sensor_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer([det_sub, pc_sub], queue_size=10,
                                                                slop=float(p('sync_slop')))
        self.sync.registerCallback(self._pair_cb)
        self.n_frames = self.n_boxes = self.n_ranged = 0
        self.times = []
        self.create_timer(10.0, self._report)
        self.get_logger().info('object fusion: %s + %s -> /perception/objects'
                               % (p('detections_topic'), p('obstacles_topic')))

    def _info_cb(self, msg):
        self.camera_info = msg

    def _image_cb(self, msg):
        if self.pub_img.get_subscription_count() == 0:
            return
        self.images[(msg.header.stamp.sec, msg.header.stamp.nanosec)] = msg
        while len(self.images) > 10:
            del self.images[min(self.images)]

    def _intrinsics(self):
        if self.camera_info is not None:
            K, D, size = lp.intrinsics_from_camera_info(self.camera_info)
            return K, D, size
        return self.base_K, self.base_D, self.calib

    def _pair_cb(self, dets, cloud):
        t0 = time.perf_counter()
        try:
            tf = self.tf_buffer.lookup_transform(self.optical, cloud.header.frame_id, rclpy.time.Time())
        except Exception as exc:
            self.get_logger().warn('no TF %s -> %s (%s)' % (cloud.header.frame_id, self.optical, exc),
                                   throttle_duration_sec=10.0)
            return
        M = cp.transform_to_matrix(tf)                       # optical <- base_footprint
        sensor_origin = mm.invert_transform(M)[:3, 3]        # camera centre in base_footprint
        K, D, (w, h) = self._intrinsics()

        f = lp.cloud_fields(cloud, ['x', 'y', 'z', 'cluster'])
        pts = np.column_stack([f['x'], f['y'], f['z']]).astype(np.float64)
        labels = f['cluster'].astype(np.int64)
        uv, depth, idx = of.project(pts, M, K, D, w, h)

        keep = [dd for dd in dets.detections
                if dd.results and dd.results[0].hypothesis.score >= self.min_score]
        boxes = [(dd.bbox.center.position.x - dd.bbox.size_x / 2, dd.bbox.center.position.y - dd.bbox.size_y / 2,
                  dd.bbox.center.position.x + dd.bbox.size_x / 2, dd.bbox.center.position.y + dd.bbox.size_y / 2)
                 for dd in keep]
        max_depth = [of.max_depth_from_box(b, dd.results[0].hypothesis.class_id, K[1, 1], self.size_gate_factor)
                     if self.size_gate else None for b, dd in zip(boxes, keep)]
        groups, how = of.associate(boxes, uv, depth, idx, labels, max_depth=max_depth, **self.assoc)

        stamp = cloud.header.stamp
        header = cloud.header
        out = Detection3DArray()
        out.header = header
        entries, obj_xyz, obj_index = [], [], []
        for i, (dd, box, g) in enumerate(zip(keep, boxes, groups)):
            hyp2 = dd.results[0].hypothesis
            x0, y0, x1, y1 = of.shrink_box(box, self.assoc['shrink'])
            in_box = int(((uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)).sum())
            entry = {'id': dd.id, 'class': hyp2.class_id, 'score': round(float(hyp2.score), 3),
                     'box': [round(float(v), 1) for v in box], 'points_in_box': in_box,
                     'range': None, 'x': None, 'y': None, 'z': None, 'n': 0, 'how': how[i]}
            obj = of.describe(pts[g], sensor_origin)
            if obj is not None:
                entry.update(range=round(obj['range'], 3), x=round(float(obj['centroid'][0]), 3),
                             y=round(float(obj['centroid'][1]), 3), z=round(float(obj['centroid'][2]), 3),
                             n=obj['n'])
                out.detections.append(self._det3d(header, dd.id, hyp2, obj))
                obj_xyz.append(pts[g])
                obj_index.append(np.full(len(g), float(i), dtype=np.float32))
            entries.append(entry)
        self.pub_obj.publish(out)
        self.pub_json.publish(String(data=json.dumps({'stamp': [stamp.sec, stamp.nanosec],
                                                      'frame': header.frame_id, 'objects': entries})))
        xyz = np.vstack(obj_xyz) if obj_xyz else np.zeros((0, 3))
        ind = np.concatenate(obj_index) if obj_index else np.zeros(0, np.float32)
        self.pub_pts.publish(self._cloud(header, xyz, ind))
        self._markers(header, entries)
        self._image(stamp, entries, K, D, M, pts, groups)

        self.times.append((time.perf_counter() - t0) * 1000.0)
        self.n_frames += 1
        self.n_boxes += len(entries)
        self.n_ranged += sum(e['range'] is not None for e in entries)

    @staticmethod
    def _det3d(header, det_id, hyp2, obj):
        d = Detection3D()
        d.header = header
        d.id = det_id
        c = (obj['min'] + obj['max']) / 2.0
        s = np.maximum(obj['max'] - obj['min'], 0.05)
        d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = map(float, c)
        d.bbox.center.orientation.w = 1.0
        d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = map(float, s)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id, hyp.hypothesis.score = hyp2.class_id, hyp2.score
        hyp.pose.pose.position.x, hyp.pose.pose.position.y, hyp.pose.pose.position.z = map(float, obj['centroid'])
        hyp.pose.pose.orientation.w = 1.0
        cov = np.zeros((6, 6))
        cov[:3, :3] = obj['cov']
        cov[3:, 3:] = np.eye(3) * 1e6                        # orientation unknown
        hyp.pose.covariance = cov.ravel().tolist()
        d.results.append(hyp)
        return d

    @staticmethod
    def _cloud(header, xyz, index):
        data = np.column_stack([xyz, index]).astype(np.float32) if len(xyz) else np.zeros((0, 4), np.float32)
        m = PointCloud2()
        m.header = header
        m.height, m.width = 1, len(data)
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(('x', 'y', 'z', 'object'))]
        m.is_bigendian, m.point_step = False, 16
        m.row_step, m.is_dense = 16 * len(data), True
        m.data = array.array('B', data.tobytes())
        return m

    def _markers(self, header, entries):
        ma = MarkerArray()
        clear = Marker()
        clear.header, clear.action = header, Marker.DELETEALL
        ma.markers.append(clear)
        for i, e in enumerate(entries):
            if e['range'] is None:
                continue
            b, g, r = box_colour(e['class'])
            m = Marker()
            m.header, m.ns, m.id, m.type, m.action = header, 'objects', i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = e['x'], e['y'], e['z']
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.5
            m.color.r, m.color.g, m.color.b, m.color.a = r / 255.0, g / 255.0, b / 255.0, 0.9
            ma.markers.append(m)
            t = Marker()
            t.header, t.ns, t.id, t.type, t.action = header, 'object_labels', i, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = e['x'], e['y'], e['z'] + 1.0
            t.scale.z = 0.5
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = '%s %.1f m' % (e['class'], e['range'])
            ma.markers.append(t)
        self.pub_mk.publish(ma)

    def _image(self, stamp, entries, K, D, M, pts, groups):
        msg = self.images.pop((stamp.sec, stamp.nanosec), None)
        if msg is None or self.pub_img.get_subscription_count() == 0:
            return
        img = lp.image_to_bgr(msg)
        for e, g in zip(entries, groups):
            col = box_colour(e['class'])
            x0, y0, x1, y1 = (int(v) for v in e['box'])
            cv2.rectangle(img, (x0, y0), (x1, y1), col if e['range'] is not None else (150, 150, 150), 2)
            if len(g):
                opt = cp.transform_points(M, pts[g])
                ok = cp.visible_mask(opt, 0.15, 2.35)
                for u, v in cp.project_pixels(opt[ok], K, D):
                    cv2.circle(img, (int(u), int(v)), 2, col, -1)
            text = ('%s %.1f m  (x %.1f, y %+.1f)' % (e['class'], e['range'], e['x'], e['y'])
                    if e['range'] is not None else '%s: no lidar' % e['class'])
            org = (x0, max(y0 - 5, 12))
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        out = Image()
        out.header = msg.header
        out.height, out.width, out.encoding, out.is_bigendian = img.shape[0], img.shape[1], 'bgr8', 0
        out.step = img.shape[1] * 3
        out.data = array.array('B', img.tobytes())
        self.pub_img.publish(out)

    def _report(self):
        if not self.n_frames:
            self.get_logger().warn('no detection + obstacle-cloud pairs in the last 10 s', throttle_duration_sec=30.0)
            return
        t = np.array(self.times)
        self.get_logger().info('%d frames, %.1f boxes/frame, %.0f %% with a lidar range, %.1f ms mean, p95 %.1f ms'
                               % (self.n_frames, self.n_boxes / self.n_frames,
                                  100.0 * self.n_ranged / max(self.n_boxes, 1), t.mean(), np.percentile(t, 95)))
        self.n_frames = self.n_boxes = self.n_ranged = 0
        self.times = []


def main(args=None):
    rclpy.init(args=args)
    node = ObjectFusionNode()
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
