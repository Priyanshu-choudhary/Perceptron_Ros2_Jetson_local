#!/usr/bin/env python3
"""Predict where each tracked object will be (roadmap Stage 7).

    /perception/tracked_objects (Stage 6, odom)
      -> per track: static/dynamic gate with hysteresis, turn-rate estimate from its
         heading history, prediction.predict() (shrinkage, CV / CTRV, physical caps,
         propagated + capped uncertainty)
      -> /perception/predicted_objects  perceptron_msgs/PredictedObjectArray (odom, the
         tracks' stamp): centre + 2D covariance at each horizon, dense path, model,
         speed and yaw rate used, footprint radius -- the Stage 8 costmap input
      -> /perception/prediction_markers  path, 2-sigma ellipse per horizon, label

Only CONFIRMED and COASTING tracks are predicted. The math and its reasons are in
prediction.py.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from perceptron_msgs.msg import PredictedObject, PredictedObjectArray, TrackedObject, TrackedObjectArray
from perceptron_perception import prediction as pr

COLOURS = {'pedestrian': (1.0, 0.3, 0.3), 'vehicle': (0.3, 0.9, 0.3), 'other': (1.0, 0.8, 0.0)}


def yaw_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class TrajectoryPredictorNode(Node):

    def __init__(self):
        super().__init__('trajectory_predictor_node')
        d = self.declare_parameter
        d('tracks_topic', '/perception/tracked_objects')
        d('horizons', [0.5, 1.0, 2.0])
        d('path_dt', 0.1)
        # on: the tracker's `moving` already needs speed > 2 sigma; a further delay only made
        # cars pulling away from a stop be predicted static (12 m error at 2 s, measured)
        d('gate_on_frames', 1)
        d('gate_off_frames', 10)
        d('yaw_window', 1.5)
        d('min_yaw_rate', 0.1)
        d('yaw_sigmas', 3.0)
        # CTRV off until headings are reliable: measured worse than CV on CARLA traffic
        # (5.2 vs 4.1 m at 2 s) because car headings from the near-face centroid are noisy
        d('use_ctrv', False)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.horizons = [float(h) for h in p('horizons')]
        self.path_dt = float(p('path_dt'))
        self.gate_args = (int(p('gate_on_frames')), int(p('gate_off_frames')))
        self.yaw_window = float(p('yaw_window'))
        self.min_yaw_rate = float(p('min_yaw_rate'))
        self.yaw_sigmas = float(p('yaw_sigmas'))
        self.use_ctrv = bool(p('use_ctrv'))
        self.gates, self.yaws, self.seen = {}, {}, {}
        self.pub = self.create_publisher(PredictedObjectArray, '/perception/predicted_objects', 10)
        self.pub_mk = self.create_publisher(MarkerArray, '/perception/prediction_markers', 10)
        self.create_subscription(TrackedObjectArray, p('tracks_topic'), self._cb, 10)
        self.counts = {pr.STATIC: 0, pr.CV: 0, pr.CTRV: 0}
        self.create_timer(10.0, self._report)
        self.get_logger().info('predicting %s at %s s' % (p('tracks_topic'), self.horizons))

    def _cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out = PredictedObjectArray()
        out.header = msg.header
        ma = MarkerArray()
        clear = Marker()
        clear.header, clear.action = msg.header, Marker.DELETEALL
        ma.markers.append(clear)
        for o in msg.objects:
            if o.state == TrackedObject.TENTATIVE:
                continue
            gate = self.gates.setdefault(o.id, pr.MotionGate(*self.gate_args))
            yaw = self.yaws.setdefault(o.id, pr.YawRateEstimator(self.yaw_window))
            self.seen[o.id] = t
            vx, vy = o.twist.twist.linear.x, o.twist.twist.linear.y
            dynamic = gate.update(bool(o.moving))
            if o.state == TrackedObject.CONFIRMED:       # coasting velocity is a prediction itself
                yaw.add(t, vx, vy)
            state = [o.pose.pose.position.x, o.pose.pose.position.y, vx, vy]
            P = np.array(o.state_covariance).reshape(4, 4)
            res = pr.predict(state, P, o.class_id, dynamic, self.horizons, self.path_dt,
                             yaw=yaw.estimate() if self.use_ctrv else (0.0, float('inf')),
                             min_yaw_rate=self.min_yaw_rate,
                             yaw_sigmas=self.yaw_sigmas)
            self.counts[res['model']] += 1
            out.objects.append(self._msg(o, res))
            self._markers(ma, msg.header, o, res)
        # forget tracks gone for 5 s
        for k in [k for k, ts in self.seen.items() if t - ts > 5.0]:
            self.gates.pop(k, None)
            self.yaws.pop(k, None)
            self.seen.pop(k, None)
        self.pub.publish(out)
        self.pub_mk.publish(ma)

    def _msg(self, o, res):
        m = PredictedObject()
        m.id, m.class_id, m.model = o.id, o.class_id, res['model']
        m.speed, m.yaw_rate, m.radius = float(res['speed']), float(res['yaw_rate']), float(res['radius'])
        m.horizons = self.horizons
        z = o.pose.pose.position.z
        for mean, S in zip(res['means'], res['covs']):
            pw = type(o.pose)()
            pw.pose.position.x, pw.pose.position.y, pw.pose.position.z = float(mean[0]), float(mean[1]), z
            pw.pose.orientation.w = 1.0
            cov = np.zeros((6, 6))
            cov[:2, :2] = S
            cov[2, 2] = 0.1
            cov[3:, 3:] = np.eye(3) * 1e6
            pw.covariance = cov.ravel().tolist()
            m.poses.append(pw)
        m.path = [Point(x=float(a), y=float(b), z=z) for a, b in res['path']]
        m.path_dt = self.path_dt
        return m

    def _markers(self, ma, header, o, res):
        r, g, b = COLOURS[pr.class_group(o.class_id)]
        z = o.pose.pose.position.z
        if res['model'] != pr.STATIC:
            line = Marker()
            line.header, line.ns, line.id, line.type, line.action = header, 'paths', o.id, Marker.LINE_STRIP, Marker.ADD
            line.points = [Point(x=float(a), y=float(bb), z=z) for a, bb in res['path']]
            line.scale.x = 0.08
            line.color.r, line.color.g, line.color.b, line.color.a = r, g, b, 1.0
            line.pose.orientation.w = 1.0
            ma.markers.append(line)
        for k, (mean, S) in enumerate(zip(res['means'], res['covs'])):
            if res['model'] == pr.STATIC and k < len(res['means']) - 1:
                continue                                  # one ellipse is enough for a static object
            vals, vecs = np.linalg.eigh(S)
            e = Marker()
            e.header, e.ns, e.id, e.type, e.action = header, 'ellipse_%d' % k, o.id, Marker.CYLINDER, Marker.ADD
            e.pose.position.x, e.pose.position.y, e.pose.position.z = float(mean[0]), float(mean[1]), z
            (e.pose.orientation.x, e.pose.orientation.y, e.pose.orientation.z,
             e.pose.orientation.w) = yaw_quaternion(math.atan2(vecs[1, 1], vecs[0, 1]))
            # 2 sigma half-axes, plus the footprint
            e.scale.x = float(2 * (2 * math.sqrt(max(vals[1], 0)) + res['radius']))
            e.scale.y = float(2 * (2 * math.sqrt(max(vals[0], 0)) + res['radius']))
            e.scale.z = 0.05
            e.color.r, e.color.g, e.color.b = r, g, b
            e.color.a = 0.45 - 0.12 * k
            ma.markers.append(e)
        t = Marker()
        t.header, t.ns, t.id, t.type, t.action = header, 'prediction_labels', o.id, Marker.TEXT_VIEW_FACING, Marker.ADD
        t.pose.position.x, t.pose.position.y = o.pose.pose.position.x, o.pose.pose.position.y
        t.pose.position.z = z + 1.9
        t.scale.z = 0.4
        t.color.r = t.color.g = t.color.b = t.color.a = 1.0
        t.text = '%s %.1f m/s' % (pr.MODEL_NAMES[res['model']], res['speed'])
        ma.markers.append(t)

    def _report(self):
        total = sum(self.counts.values())
        if not total:
            self.get_logger().warn('no confirmed tracks to predict', throttle_duration_sec=30.0)
            return
        self.get_logger().info('predictions: %s' % ', '.join('%s %.0f %%' % (pr.MODEL_NAMES[k], 100.0 * v / total)
                                                              for k, v in self.counts.items()))
        self.counts = {pr.STATIC: 0, pr.CV: 0, pr.CTRV: 0}


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPredictorNode()
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
