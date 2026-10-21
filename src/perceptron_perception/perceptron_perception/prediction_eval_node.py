#!/usr/bin/env python3
"""Score trajectory_predictor_node against CARLA's future ground truth (roadmap Stage 7).

    /perception/tracked_objects + /perception/predicted_objects (odom, same stamp)
    + /carla/ground_truth/objects (carla_world, every tick)
        -> predictions matched to actors by position at the time they were made, then
           scored when the truth for t + 0.5 / 1 / 2 s arrives: error, 95 % ellipse
           coverage, and the "stay" / "raw cv" baselines. Definitions: prediction_eval.py.
"""

import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from perceptron_msgs.msg import PredictedObjectArray, TrackedObject, TrackedObjectArray
from perceptron_navigation import camera_projection as cp
from perceptron_perception import eval_report as er
from perceptron_perception import prediction_eval as pe
from perceptron_perception import tracking_eval as te


class PredictionEvalNode(Node):

    def __init__(self):
        super().__init__('prediction_eval_node')
        self.declare_parameter('world_frame', 'carla_world')
        self.declare_parameter('horizons', [0.5, 1.0, 2.0])
        self.declare_parameter('report_period', 15.0)
        self.declare_parameter('summary_file', '')       # JSON for the scenario suite
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.world = p('world_frame')
        self.horizons = [float(h) for h in p('horizons')]
        self.scores = pe.PredictionScores(self.horizons)
        self.pending, self.ready, self.truth_now = {}, [], {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(TrackedObjectArray, '/perception/tracked_objects', lambda m: self._add('trk', m), 10)
        self.create_subscription(PredictedObjectArray, '/perception/predicted_objects', lambda m: self._add('pred', m), 10)
        self.create_subscription(TrackedObjectArray, '/carla/ground_truth/objects', self._gt_cb, 10)
        self.create_timer(0.05, self._drain)
        self.create_timer(float(p('report_period')), self._report)

    @staticmethod
    def _key(stamp):
        return (stamp.sec, stamp.nanosec)

    def _gt_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        truth = {o.id: {'class': o.class_id,
                        'xy': np.array([o.pose.pose.position.x, o.pose.pose.position.y]),
                        'speed': float(np.hypot(o.twist.twist.linear.x, o.twist.twist.linear.y))}
                 for o in msg.objects}
        self.truth_now[self._key(msg.header.stamp)] = truth
        while len(self.truth_now) > 20:
            del self.truth_now[min(self.truth_now)]
        self.scores.add_truth(t, {a: v['xy'] for a, v in truth.items()})

    def _add(self, kind, msg):
        k = self._key(msg.header.stamp)
        e = self.pending.setdefault(k, {})
        e[kind] = msg
        if len(e) == 2:
            del self.pending[k]
            self.ready.append(e)
        if len(self.pending) > 40:
            for kk in sorted(self.pending)[:-40]:
                del self.pending[kk]

    def _drain(self):
        # never wait inside a callback (see object_tracker_node): score once TF is there
        keep = []
        for e in self.ready:
            when = Time.from_msg(e['trk'].header.stamp)
            truth = self.truth_now.get(self._key(e['trk'].header.stamp))
            if truth is not None and self.tf_buffer.can_transform(self.world, e['trk'].header.frame_id, when):
                self._register(e, when, truth)
            else:
                keep.append(e)
        self.ready = keep[-40:]

    def _register(self, e, when, truth):
        M = cp.transform_to_matrix(self.tf_buffer.lookup_transform(self.world, e['trk'].header.frame_id, when))
        R = M[:2, :2]
        tracks, by_id = [], {}
        for o in e['trk'].objects:
            if o.state == TrackedObject.TENTATIVE:
                continue
            pos = cp.transform_points(M, np.array([[o.pose.pose.position.x, o.pose.pose.position.y, 0.0]]))[0, :2]
            vel = R @ np.array([o.twist.twist.linear.x, o.twist.twist.linear.y])
            tr = {'id': o.id, 'class': o.class_id, 'xy': pos, 'v': vel}
            tracks.append(tr)
            by_id[o.id] = tr
        preds = {p.id: p for p in e['pred'].objects}
        items = []
        for tr, actor in te.match_by_position(tracks, truth):
            p = preds.get(tr['id'])
            if p is None:
                continue
            means = np.array([cp.transform_points(M, np.array([[q.pose.position.x, q.pose.position.y, 0.0]]))[0, :2]
                              for q in p.poses])
            covs = np.array([R @ np.array(q.covariance).reshape(6, 6)[:2, :2] @ R.T for q in p.poses])
            items.append({'actor': actor, 'cls': truth[actor]['class'], 'moving': truth[actor]['speed'] >= 0.2,
                          'model': p.model, 'means': means, 'covs': covs, 'pos': tr['xy'], 'vel': tr['v']})
        t = e['trk'].header.stamp.sec + e['trk'].header.stamp.nanosec * 1e-9
        self.scores.add_predictions(t, items)

    def _report(self):
        self._write_summary()
        s = self.scores.summary()
        if not s:
            self.get_logger().warn('nothing scored yet (predictions need %.1f s of future truth)'
                                   % max(self.horizons), throttle_duration_sec=30.0)
            return
        self.get_logger().info('\n' + pe.format_summary(s, self.scores.models, 'prediction vs future ground truth, '
                                                        'since start') +
                               '\n' + pe.format_by_model(self.scores.by_model, self.horizons[-1]))

    def _write_summary(self):
        path = self.get_parameter('summary_file').value
        if path and self.scores.summary():
            er.write_json(path, {'node': self.get_name(), 'summary': self.scores.summary()})

    def destroy_node(self):
        self._write_summary()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PredictionEvalNode()
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
