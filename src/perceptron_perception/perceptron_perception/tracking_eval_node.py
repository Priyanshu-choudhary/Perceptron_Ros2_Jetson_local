#!/usr/bin/env python3
"""Score object_tracker_node against CARLA ground truth (roadmap Stage 6).

    /perception/tracked_objects (odom) + /carla/ground_truth/objects (carla_world, true
    velocity), paired by stamp; tracks are moved into carla_world through TF (velocity
    rotated with them), so in ekf mode the score includes the EKF's own error.
        -> every report_period: last period and since start, per class: ID switches,
           swapped tracks, position error, velocity error static / moving, speed and
           heading error. Definitions: tracking_eval.py.
"""

import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from perceptron_msgs.msg import TrackedObject, TrackedObjectArray
from perceptron_navigation import camera_projection as cp
from perceptron_perception import eval_report as er
from perceptron_perception import tracking_eval as te


class TrackingEvalNode(Node):

    def __init__(self):
        super().__init__('tracking_eval_node')
        self.declare_parameter('world_frame', 'carla_world')
        self.declare_parameter('report_period', 10.0)
        self.declare_parameter('summary_file', '')       # JSON for the scenario suite
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.world = p('world_frame')
        self.period, self.total = te.TrackingScores(), te.TrackingScores()
        self.pending = {}
        self.ready = []
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(TrackedObjectArray, '/perception/tracked_objects', lambda m: self._add('trk', m), 10)
        self.create_subscription(TrackedObjectArray, '/carla/ground_truth/objects', lambda m: self._add('gt', m), 10)
        self.create_timer(float(p('report_period')), self._report)
        self.create_timer(0.05, self._drain)

    def _add(self, kind, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        e = self.pending.setdefault(key, {})
        e[kind] = msg
        if len(e) == 2:
            del self.pending[key]
            self.ready.append(e)
        if len(self.pending) > 40:
            for k in sorted(self.pending)[:-40]:
                del self.pending[k]

    def _drain(self):
        # Never wait inside a callback: it would starve this node's /tf subscription (see
        # object_tracker_node). Score each pair once its transform exists; keep at most 40.
        keep = []
        for e in self.ready:
            when = Time.from_msg(e['trk'].header.stamp)
            if self.tf_buffer.can_transform(self.world, e['trk'].header.frame_id, when):
                self._evaluate(e, when)
            else:
                keep.append(e)
        self.ready = keep[-40:]

    def _evaluate(self, e, when):
        trk = e['trk']
        M = cp.transform_to_matrix(self.tf_buffer.lookup_transform(self.world, trk.header.frame_id, when))
        truth = {}
        for o in e['gt'].objects:
            pp, v = o.pose.pose.position, o.twist.twist.linear
            truth[o.id] = {'class': o.class_id, 'xy': np.array([pp.x, pp.y]),
                                            'v': np.array([v.x, v.y])}
        tracks = []
        for o in trk.objects:
            if o.state == TrackedObject.TENTATIVE:
                continue
            pp, v = o.pose.pose.position, o.twist.twist.linear
            xy = cp.transform_points(M, np.array([[pp.x, pp.y, pp.z]]))[0, :2]
            vw = M[:3, :3] @ np.array([v.x, v.y, 0.0])
            tracks.append({'id': o.id, 'class': o.class_id, 'xy': xy, 'v': vw[:2]})
        for s in (self.period, self.total):
            s.add(tracks, truth)

    def _report(self):
        self._write_summary()
        if not self.period.frames:
            self.get_logger().warn('nothing scored yet: needs carla_bridge, carla_tracking.launch.py',
                                   throttle_duration_sec=30.0)
            return
        self.get_logger().info('\n' + te.format_summary(self.period.summary(), 'tracking, last period') +
                               '\n' + te.format_summary(self.total.summary(), 'tracking, since start'))
        self.period = te.TrackingScores()

    def _write_summary(self):
        path = self.get_parameter('summary_file').value
        if path and self.total.frames:
            er.write_json(path, {'node': self.get_name(), 'summary': self.total.summary()})

    def destroy_node(self):
        self._write_summary()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrackingEvalNode()
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
