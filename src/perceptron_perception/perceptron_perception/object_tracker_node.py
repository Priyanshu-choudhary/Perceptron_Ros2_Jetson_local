#!/usr/bin/env python3
"""Track Stage 5 objects over time in odom: stable ids and velocities (roadmap Stage 6).

    /perception/objects (Detection3DArray, base_footprint, from object_fusion_node)
      --TF odom <- base_footprint at each message's stamp (ego-motion compensation:
        the EKF's odom in ekf mode, ground truth in gt mode)-->
      tracker.py: constant-velocity Kalman per track, Mahalanobis gate + Hungarian,
                  tentative -> confirmed -> coasting -> deleted
      -> /perception/tracked_objects  perceptron_msgs/TrackedObjectArray in odom
      -> /perception/track_markers    MarkerArray: box, velocity arrow, "#id class speed"

TIMING: in ekf mode the EKF's odom -> base_footprint arrives a tick AFTER the detections it
must place. Waiting for it inside a callback blocks this node's own /tf subscription (one
executor thread), so the buffer never catches up. Messages are queued instead and a timer
processes each as soon as its transform exists, never blocking; one that cannot be placed
within max_tf_wait seconds is dropped.

Only tracks with state CONFIRMED or COASTING should be planned around; TENTATIVE ones are
published for inspection. source_detection_id carries the detector's id through for the
evaluator and is never used for tracking.
"""

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from perceptron_msgs.msg import TrackedObject, TrackedObjectArray
from perceptron_navigation import camera_projection as cp
from perceptron_perception import tracker as tk

COLOURS = {'pedestrian': (1.0, 0.25, 0.25), 'vehicle': (0.3, 0.9, 0.3), 'other': (1.0, 0.8, 0.0)}


class ObjectTrackerNode(Node):

    def __init__(self):
        super().__init__('object_tracker_node')
        d = self.declare_parameter
        d('objects_topic', '/perception/objects')
        d('tracking_frame', 'odom')
        d('sensor_frame', 'camera_optical_link')
        d('max_tf_wait', 1.0)
        d('gate_chi2', 9.21)
        d('confirm_hits', 3)
        d('tentative_misses', 2)
        d('max_coast_time', 1.0)
        d('min_moving_speed', 0.3)
        # per class group (pedestrian / vehicle / other): see tracker.DEFAULT_CLASS_PARAMS
        for g, cp_ in tk.DEFAULT_CLASS_PARAMS.items():
            for k, v in cp_.items():
                d('%s.%s' % (g, k), float(v))
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        class_params = {g: {k: float(p('%s.%s' % (g, k))) for k in cp_}
                        for g, cp_ in tk.DEFAULT_CLASS_PARAMS.items()}
        self.frame = p('tracking_frame')
        self.sensor_frame = p('sensor_frame')
        self.max_tf_wait = float(p('max_tf_wait'))
        self.tracker_args = dict(gate_chi2=float(p('gate_chi2')), confirm_hits=int(p('confirm_hits')),
                                 tentative_misses=int(p('tentative_misses')),
                                 max_coast_time=float(p('max_coast_time')),
                                 min_speed=float(p('min_moving_speed')), class_params=class_params)
        self.tracker = tk.Tracker(**self.tracker_args)
        self.last_t = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(TrackedObjectArray, '/perception/tracked_objects', 10)
        self.pub_mk = self.create_publisher(MarkerArray, '/perception/track_markers', 10)
        self.queue = []
        self.create_subscription(Detection3DArray, p('objects_topic'), self.queue.append, 20)
        self.create_timer(0.01, self._drain)
        self.n_frames = self.n_dropped = 0
        self.create_timer(10.0, self._report)
        self.get_logger().info('tracking %s in %s' % (p('objects_topic'), self.frame))

    def _drain(self):
        """Process queued detections, oldest first, once their transform is available."""
        while self.queue:
            msg = self.queue[0]
            when = Time.from_msg(msg.header.stamp)
            if (self.tf_buffer.can_transform(self.frame, msg.header.frame_id, when) and
                    self.tf_buffer.can_transform(self.frame, self.sensor_frame, when)):
                self.queue.pop(0)
                self._process(msg, when)
                continue
            newest = self.queue[-1].header.stamp
            age = (newest.sec + newest.nanosec * 1e-9) - when.nanoseconds * 1e-9
            if age > self.max_tf_wait:
                self.queue.pop(0)
                self.n_dropped += 1
                self.get_logger().warn('no TF %s <- %s for a detection %.2f s old; dropped'
                                       % (self.frame, msg.header.frame_id, age), throttle_duration_sec=5.0)
                continue
            return

    def _process(self, msg, when):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if self.last_t is not None and t < self.last_t - 1.0:
            self.get_logger().warn('time jumped back %.1f s (simulation restarted?): tracker reset'
                                   % (self.last_t - t))
            self.tracker = tk.Tracker(**self.tracker_args)
        self.last_t = t
        M = cp.transform_to_matrix(self.tf_buffer.lookup_transform(self.frame, msg.header.frame_id, when))
        S = cp.transform_to_matrix(self.tf_buffer.lookup_transform(self.frame, self.sensor_frame, when))
        R2 = M[:2, :2]
        sensor_xy = S[:2, 3]
        dets = []
        for d in msg.detections:
            if not d.results:
                continue
            h = d.results[0]
            pos = h.pose.pose.position
            xyz = cp.transform_points(M, np.array([[pos.x, pos.y, pos.z]]))[0]
            cov = np.array(h.pose.covariance).reshape(6, 6)[:2, :2]
            dets.append(self.tracker.make_detection(
                xyz[:2], xyz[2], h.hypothesis.class_id, R2 @ cov @ R2.T,
                (d.bbox.size.x, d.bbox.size.y, d.bbox.size.z), d.id, sensor_xy))
        tracks = self.tracker.step(dets, t)
        self._publish(msg.header.stamp, tracks)
        self.n_frames += 1

    def _publish(self, stamp, tracks):
        out = TrackedObjectArray()
        out.header.stamp, out.header.frame_id = stamp, self.frame
        ma = MarkerArray()
        clear = Marker()
        clear.header = out.header
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for tr in tracks:
            o = TrackedObject()
            o.id, o.class_id, o.class_confidence = tr.id, tr.class_id, float(tr.class_confidence)
            o.state = tr.state
            o.moving = bool(tr.moving(self.tracker.min_speed))
            o.pose.pose.position.x, o.pose.pose.position.y = float(tr.x[0]), float(tr.x[1])
            o.pose.pose.position.z = float(tr.z)
            o.pose.pose.orientation.w = 1.0
            pc = np.zeros((6, 6))
            pc[:2, :2] = tr.P[:2, :2]
            pc[2, 2] = 0.1
            pc[3:, 3:] = np.eye(3) * 1e6
            o.pose.covariance = pc.ravel().tolist()
            o.twist.twist.linear.x, o.twist.twist.linear.y = float(tr.x[2]), float(tr.x[3])
            tc = np.zeros((6, 6))
            tc[:2, :2] = tr.P[2:, 2:]
            tc[2, 2] = 1e6
            tc[3:, 3:] = np.eye(3) * 1e6
            o.twist.covariance = tc.ravel().tolist()
            o.size.x, o.size.y, o.size.z = (float(v) for v in tr.size)
            o.state_covariance = tr.P.ravel().astype(float).tolist()
            o.age = float(tr.t - tr.t_first)
            o.hits, o.misses = int(tr.hits), int(tr.misses)
            o.source_detection_id = tr.source
            out.objects.append(o)
            if tr.state != tk.TENTATIVE:
                self._markers(ma, out.header, tr, o)
        self.pub.publish(out)
        self.pub_mk.publish(ma)

    @staticmethod
    def _markers(ma, header, tr, o):
        r, g, b = COLOURS[tr.group]
        alpha = 0.8 if tr.state == tk.CONFIRMED else 0.35
        box = Marker()
        box.header, box.ns, box.id, box.type, box.action = header, 'tracks', tr.id, Marker.CYLINDER, Marker.ADD
        box.pose = o.pose.pose
        size = max(0.4, min(4.5, float(max(tr.size[0], tr.size[1]))))
        box.scale.x = box.scale.y = size
        box.scale.z = 0.2
        box.color.r, box.color.g, box.color.b, box.color.a = r, g, b, alpha
        ma.markers.append(box)
        if tr.speed > 0.1:
            arrow = Marker()
            arrow.header, arrow.ns, arrow.id, arrow.type = header, 'velocity', tr.id, Marker.ARROW
            arrow.action = Marker.ADD
            p0 = Point(x=float(tr.x[0]), y=float(tr.x[1]), z=float(tr.z))
            p1 = Point(x=float(tr.x[0] + tr.x[2]), y=float(tr.x[1] + tr.x[3]), z=float(tr.z))  # 1 s ahead
            arrow.points = [p0, p1]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.1, 0.25, 0.25
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = r, g, b, 1.0
            ma.markers.append(arrow)
        text = Marker()
        text.header, text.ns, text.id, text.type = header, 'track_labels', tr.id, Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x, text.pose.position.y = float(tr.x[0]), float(tr.x[1])
        text.pose.position.z = float(tr.z) + 1.4
        text.scale.z = 0.45
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = '#%d %s %.1f m/s%s' % (tr.id, tr.class_id, tr.speed, '' if tr.state == tk.CONFIRMED else ' (coast)')
        ma.markers.append(text)

    def _report(self):
        if not self.n_frames:
            self.get_logger().warn('no /perception/objects received', throttle_duration_sec=30.0)
            return
        tr = self.tracker.tracks
        live = [t for t in tr if t.state != tk.TENTATIVE]
        self.get_logger().info('%d frames (%d dropped without TF); now %d confirmed/coasting tracks '
                               '(%d moving), %d tentative'
                               % (self.n_frames, self.n_dropped, len(live),
                                  sum(t.moving(self.tracker.min_speed) for t in live), len(tr) - len(live)))
        self.n_frames = self.n_dropped = 0


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackerNode()
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
