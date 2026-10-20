#!/usr/bin/env python3
"""3D lidar preprocessing (roadmap Stage 4): filter, ground, clusters, scan.

    /points (lidar frame) --range--> base_footprint --self box--> voxel
        -> /perception/points_ground      ground points            (PointCloud2)
        -> /perception/points_obstacles   x,y,z,cluster (-1 = none) (PointCloud2)
        -> /perception/clusters           one box per cluster      (vision_msgs/Detection3DArray)
        -> /perception/cluster_markers    the same, for RViz       (MarkerArray)
        -> /perception/scan               nearest obstacle per bearing (LaserScan)
        -> /perception/lidar_preprocess_stats  [ms, n_in, n_kept, n_ground, n_obstacles,
                                                n_clusters, plane_tilt_deg, plane_z]

The geometry is in cloud_processing.py. Everything is published in
target_frame (base_footprint) with the cloud's own stamp. Cluster ids are per
frame -- they are NOT track ids; persistent ids are Stage 6.

The lidar -> base_footprint transform is static, so the latest one is used.
plane_tilt_deg / plane_z are the fitted ground plane in base_footprint: with a
correct lidar extrinsic and a flat road they sit near 0 (roadmap task 4.1).
"""

import array
import time

import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from perceptron_navigation import camera_projection as cp
from perceptron_perception import cloud_processing as cpr
from perceptron_perception import lidar_projection as lp


def make_cloud(header, xyz, extra=None):
    """PointCloud2 with float32 x, y, z (+ one optional float32 field)."""
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    names = ['x', 'y', 'z']
    if extra is not None:
        names.append(extra[0])
        cols.append(extra[1])
    data = np.column_stack(cols).astype(np.float32)
    m = PointCloud2()
    m.header = header
    m.height, m.width = 1, len(data)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate(names)]
    m.is_bigendian, m.point_step = False, 4 * len(names)
    m.row_step, m.is_dense = m.point_step * len(data), True
    m.data = array.array('B', data.tobytes())
    return m


def colour(k):
    h = (k * 0.61803398875) % 1.0
    i, f = int(h * 6), h * 6 - int(h * 6)
    q, t = 1.0 - f, f
    rgb = [(1, t, 0), (q, 1, 0), (0, 1, t), (0, q, 1), (t, 0, 1), (1, 0, q)][i % 6]
    return tuple(float(c) for c in rgb)        # Marker colours must be float, not int


class LidarPreprocessorNode(Node):

    def __init__(self):
        super().__init__('lidar_preprocessor_node')
        d = self.declare_parameter
        d('cloud_topic', '/points')
        d('target_frame', 'base_footprint')
        d('min_range', 0.5)
        d('max_range', 50.0)
        # Points of the robot (or the sim host vehicle) itself, in target_frame:
        # [x0, x1, y0, y1, z0, z1]. self_box_enabled false skips it.
        d('self_box_enabled', False)
        d('self_box', [-0.3, 0.3, -0.3, 0.3, -0.5, 1.0])
        d('voxel', 0.08)
        d('obstacle_z_max', 2.5)
        d('plane_band', 0.6)
        d('plane_threshold', 0.15)
        d('ransac_threshold', 0.05)
        d('cell', 0.5)
        d('cell_height', 0.12)
        d('cell_plane_tol', 0.3)
        d('cluster_tolerance', 0.5)
        d('cluster_min_points', 6)
        d('publish_scan', True)
        d('scan_z_min', 0.1)
        d('scan_z_max', 1.5)
        d('scan_angle_increment_deg', 0.5)
        d('scan_range_max', 30.0)

        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.target = p('target_frame')
        self.min_r, self.max_r = float(p('min_range')), float(p('max_range'))
        self.self_box = [float(v) for v in p('self_box')] if p('self_box_enabled') else None
        self.voxel = float(p('voxel'))
        self.z_max = float(p('obstacle_z_max'))
        self.ground_args = dict(plane_band=float(p('plane_band')), plane_threshold=float(p('plane_threshold')),
                                ransac_threshold=float(p('ransac_threshold')), cell=float(p('cell')),
                                cell_height=float(p('cell_height')), cell_plane_tol=float(p('cell_plane_tol')))
        self.tol = float(p('cluster_tolerance'))
        self.min_pts = int(p('cluster_min_points'))
        self.scan_on = bool(p('publish_scan'))
        self.scan_z = (float(p('scan_z_min')), float(p('scan_z_max')))
        self.scan_inc = float(np.radians(p('scan_angle_increment_deg')))
        self.scan_rmax = float(p('scan_range_max'))
        self.rng = np.random.default_rng(0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.pub_ground = self.create_publisher(PointCloud2, '/perception/points_ground', qos)
        self.pub_obst = self.create_publisher(PointCloud2, '/perception/points_obstacles', qos)
        self.pub_det = self.create_publisher(Detection3DArray, '/perception/clusters', 10)
        self.pub_mk = self.create_publisher(MarkerArray, '/perception/cluster_markers', 10)
        self.pub_scan = self.create_publisher(LaserScan, '/perception/scan', qos)
        self.pub_stats = self.create_publisher(Float32MultiArray, '/perception/lidar_preprocess_stats', 10)
        self.create_subscription(PointCloud2, p('cloud_topic'), self._cloud_cb, qos)

        self.times = []
        self.create_timer(10.0, self._report)
        self.get_logger().info('lidar preprocessing %s -> %s, self box %s'
                               % (p('cloud_topic'), self.target, self.self_box))

    def _cloud_cb(self, msg):
        t0 = time.perf_counter()
        try:
            tf = self.tf_buffer.lookup_transform(self.target, msg.header.frame_id, rclpy.time.Time())
        except Exception as exc:
            self.get_logger().warn('no TF %s -> %s (%s)' % (msg.header.frame_id, self.target, exc),
                                   throttle_duration_sec=10.0)
            return
        raw = lp.cloud_to_xyz(msg)
        n_in = len(raw)
        xyz = raw[cpr.range_mask(raw, self.min_r, self.max_r)]
        xyz = cp.transform_points(cp.transform_to_matrix(tf), xyz)
        if self.self_box is not None:
            xyz = xyz[~cpr.box_mask(xyz, self.self_box)]
        xyz = xyz[cpr.voxel_downsample(xyz, self.voxel)]

        ground, plane = cpr.segment_ground(xyz, rng=self.rng, **self.ground_args)
        obst = xyz[~ground & (xyz[:, 2] < self.z_max)]
        labels = cpr.cluster_2d(obst[:, :2], self.tol, self.min_pts)
        boxes = cpr.cluster_boxes(obst, labels)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        header = msg.header
        header.frame_id = self.target
        self.pub_ground.publish(make_cloud(header, xyz[ground]))
        self.pub_obst.publish(make_cloud(header, obst, ('cluster', labels.astype(np.float32))))
        self._publish_clusters(header, boxes)
        if self.scan_on:
            self._publish_scan(header, obst)

        tilt = cpr.plane_tilt_deg(plane[0]) if plane is not None else float('nan')
        z0 = cpr.plane_height_at_origin(plane) if plane is not None else float('nan')
        self.pub_stats.publish(Float32MultiArray(data=[
            float(elapsed_ms), float(n_in), float(len(xyz)), float(ground.sum()), float(len(obst)),
            float(len(boxes)), float(tilt), float(z0)]))
        self.times.append(elapsed_ms)

    def _publish_clusters(self, header, boxes):
        det = Detection3DArray()
        det.header = header
        ma = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for b in boxes:
            centre = (b['min'] + b['max']) / 2.0
            size = np.maximum(b['max'] - b['min'], 0.05)
            d = Detection3D()
            d.header = header
            d.id = str(b['id'])
            d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = map(float, centre)
            d.bbox.center.orientation.w = 1.0
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = map(float, size)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = 'cluster', 1.0
            hyp.pose.pose.position.x, hyp.pose.pose.position.y, hyp.pose.pose.position.z = map(float, b['centroid'])
            hyp.pose.pose.orientation.w = 1.0
            d.results.append(hyp)
            det.detections.append(d)

            r, g, bl = colour(b['id'])
            m = Marker()
            m.header, m.ns, m.id, m.type, m.action = header, 'clusters', b['id'], Marker.CUBE, Marker.ADD
            m.pose = d.bbox.center
            m.scale.x, m.scale.y, m.scale.z = map(float, size)
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, bl, 0.35
            ma.markers.append(m)
            t = Marker()
            t.header, t.ns, t.id, t.type, t.action = header, 'labels', b['id'], Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y = float(centre[0]), float(centre[1])
            t.pose.position.z = float(b['max'][2] + 0.3)
            t.scale.z = 0.4
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = '%d (%d)' % (b['id'], b['n'])
            ma.markers.append(t)
        self.pub_det.publish(det)
        self.pub_mk.publish(ma)

    def _publish_scan(self, header, obst):
        s = LaserScan()
        s.header = header
        s.angle_min, s.angle_max = -np.pi, np.pi - self.scan_inc
        s.angle_increment = self.scan_inc
        s.range_min, s.range_max = self.min_r, self.scan_rmax
        s.ranges = cpr.to_laserscan(obst, s.angle_min, s.angle_max, self.scan_inc, s.range_min,
                                    s.range_max, self.scan_z[0], self.scan_z[1]).tolist()
        self.pub_scan.publish(s)

    def _report(self):
        if not self.times:
            self.get_logger().warn('no clouds processed in the last 10 s', throttle_duration_sec=30.0)
            return
        t = np.array(self.times)
        self.get_logger().info('%d clouds, processing mean %.1f ms, p95 %.1f ms, max %.1f ms'
                               % (len(t), t.mean(), np.percentile(t, 95), t.max()))
        self.times = []


def main(args=None):
    rclpy.init(args=args)
    node = LidarPreprocessorNode()
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
