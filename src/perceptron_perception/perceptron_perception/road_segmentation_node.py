#!/usr/bin/env python3
"""Road, sidewalk and lane-line map from semantic segmentation, for RViz and later planning.

    /carla/camera/semantic (mono8 class ids) + /carla/camera/depth (32FC1, optional)
    + /carla/camera/camera_info + TF
      -> /perception/road_cloud   PointCloud2 x, y, z, rgb in odom: the rolling map of the
                                  latest label per 0.1 m cell (green road, magenta sidewalk,
                                  orange markings)
      -> /perception/road_stats   Float32MultiArray [ms, road, sidewalk, line cells]

projection:=depth uses the depth image; projection:=ground intersects pixel rays with the
ground plane, which is what a segmentation model on the real rover's camera would use.
The class ids per label are parameters, so a model with another label set only changes
road_ids / sidewalk_ids / line_ids. Math in road_segmentation.py.
"""

import array
import time

import message_filters
import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray

from perceptron_navigation import camera_projection as cp
from perceptron_perception import lidar_projection as lp
from perceptron_perception import road_segmentation as rs


class RoadSegmentationNode(Node):

    def __init__(self):
        super().__init__('road_segmentation_node')
        d = self.declare_parameter
        d('semantic_topic', '/carla/camera/semantic')
        d('depth_topic', '/carla/camera/depth')
        d('camera_info_topic', '/carla/camera/camera_info')
        d('projection', 'depth')               # depth | ground
        d('global_frame', 'odom')
        d('robot_frame', 'base_footprint')
        d('road_ids', [1])                     # CARLA 0.9.15 tags
        d('sidewalk_ids', [2])
        d('line_ids', [24])
        d('stride', 4)                         # road / sidewalk pixel subsampling
        d('max_range', 25.0)
        d('size', 40.0)
        d('resolution', 0.1)
        d('memory', 60.0)                      # s a cell keeps its label unseen
        d('max_rate', 10.0)                    # Hz of images used
        d('publish_period', 0.2)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.frame, self.robot_frame = p('global_frame'), p('robot_frame')
        self.use_depth = p('projection') == 'depth'
        self.class_ids = {rs.ROAD: list(p('road_ids')), rs.SIDEWALK: list(p('sidewalk_ids')),
                          rs.LINE: list(p('line_ids'))}
        self.stride, self.max_range = int(p('stride')), float(p('max_range'))
        self.min_dt = 1.0 / float(p('max_rate'))
        self.map = rs.RoadMap(size=float(p('size')), res=float(p('resolution')), memory=float(p('memory')))
        self.K = None
        self.queue, self.last_t, self.times = [], -1e9, []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CameraInfo, p('camera_info_topic'), self._info_cb, 10)
        if self.use_depth:
            sem = message_filters.Subscriber(self, Image, p('semantic_topic'), qos_profile=qos)
            dep = message_filters.Subscriber(self, Image, p('depth_topic'), qos_profile=qos)
            self.sync = message_filters.TimeSynchronizer([sem, dep], 10)
            self.sync.registerCallback(self._enqueue)
        else:
            self.create_subscription(Image, p('semantic_topic'), lambda m: self._enqueue(m, None), qos)
        self.pub = self.create_publisher(PointCloud2, '/perception/road_cloud', 2)
        self.pub_stats = self.create_publisher(Float32MultiArray, '/perception/road_stats', 10)
        self.create_timer(0.01, self._drain)
        self.create_timer(float(p('publish_period')), self._publish)
        self.create_timer(10.0, self._report)
        self.get_logger().info('road map from %s, %s projection, %.1f m cells'
                               % (p('semantic_topic'), p('projection'), p('resolution')))

    def _info_cb(self, msg):
        self.K = lp.intrinsics_from_camera_info(msg)[0]

    def _enqueue(self, sem, depth):
        t = sem.header.stamp.sec + sem.header.stamp.nanosec * 1e-9
        if t - self.last_t < self.min_dt - 1e-3:
            return
        self.last_t = t
        self.queue.append((sem, depth))
        del self.queue[:-5]

    def _drain(self):
        while self.queue and self.K is not None:
            sem, depth = self.queue[0]
            when = Time.from_msg(sem.header.stamp)
            cam = sem.header.frame_id
            if not (self.tf_buffer.can_transform(self.frame, cam, when) and
                    self.tf_buffer.can_transform(self.frame, self.robot_frame, when)):
                newest = self.queue[-1][0].header.stamp
                if newest.sec + newest.nanosec * 1e-9 - when.nanoseconds * 1e-9 > 1.0:
                    self.queue.pop(0)
                    continue
                return
            self.queue.pop(0)
            self._process(sem, depth, when)

    def _process(self, sem, depth, when):
        t0 = time.perf_counter()
        t = when.nanoseconds * 1e-9
        cam = sem.header.frame_id
        M_odom_opt = cp.transform_to_matrix(self.tf_buffer.lookup_transform(self.frame, cam, when))
        robot = cp.transform_to_matrix(self.tf_buffer.lookup_transform(self.frame, self.robot_frame, when))
        ids = np.frombuffer(sem.data, np.uint8).reshape(sem.height, sem.step)[:, :sem.width]
        labels = rs.label_image(ids, self.class_ids)
        v, u = rs.select_pixels(labels, self.stride)
        lab = labels[v, u]
        if self.use_depth:
            z = lp.depth_to_array(depth)[v, u].astype(np.float64)
            ok = (z > 0.1) & (z < self.max_range)
            xyz = cp.transform_points(M_odom_opt, rs.backproject_depth(u[ok], v[ok], z[ok], self.K))
            lab = lab[ok]
        else:
            M_base_opt = np.linalg.inv(robot) @ M_odom_opt
            pts, ok = rs.ground_intersect(u.astype(float), v.astype(float), self.K, M_base_opt)
            ok &= np.hypot(pts[:, 0] - M_base_opt[0, 3], pts[:, 1] - M_base_opt[1, 3]) < self.max_range
            xyz, lab = cp.transform_points(robot, pts[ok]), lab[ok]
        self.map.recenter(robot[:2, 3])
        self.map.insert(xyz, lab, t)
        self.map.expire(t)
        self.stamp = sem.header.stamp
        self.times.append((time.perf_counter() - t0) * 1000.0)

    def _publish(self):
        if self.map.origin is None or not hasattr(self, 'stamp'):
            return
        xyz, lab = self.map.points()
        buf = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
        buf['x'], buf['y'], buf['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2] + 0.02
        buf['rgb'] = rs.packed_rgb(lab)
        m = PointCloud2()
        m.header.stamp, m.header.frame_id = self.stamp, self.frame
        m.height, m.width = 1, len(buf)
        m.fields = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
                    for k, n in enumerate(('x', 'y', 'z', 'rgb'))]
        m.is_bigendian, m.point_step, m.row_step, m.is_dense = False, 16, 16 * len(buf), True
        m.data = array.array('B', buf.tobytes())
        self.pub.publish(m)
        c = self.map.counts()
        self.pub_stats.publish(Float32MultiArray(data=[
            float(self.times[-1]) if self.times else 0.0,
            float(c['road']), float(c['sidewalk']), float(c['road_line'])]))

    def _report(self):
        if not self.times:
            self.get_logger().warn('no semantic images processed (bridge running without --no-semantic? camera_info?)',
                                   throttle_duration_sec=30.0)
            return
        a = np.array(self.times)
        c = self.map.counts()
        self.get_logger().info('%d images, %.1f ms mean; cells road %d, sidewalk %d, road_line %d'
                               % (len(a), a.mean(), c['road'], c['sidewalk'], c['road_line']))
        self.times = []


def main(args=None):
    rclpy.init(args=args)
    node = RoadSegmentationNode()
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
