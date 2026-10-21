#!/usr/bin/env python3
"""Score costmap_node against CARLA ground truth (roadmap Stage 8).

    /costmap/combined, /costmap/obstacles, /costmap/dynamic_risk, /costmap/stats
    + /carla/ground_truth/objects + /carla/ground_truth/lidar_semantic + TF
        -> every report_period (since start): obstacle recall per class and motion, false
           obstacles, dynamic-risk coverage of where moving actors really went 1 s / 2 s
           later, flicker, latency. Definitions: costmap_eval.py.
"""

import json
import math

import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String

from perceptron_msgs.msg import TrackedObjectArray
from perceptron_navigation import camera_projection as cp
from perceptron_perception import cloud_eval as ce
from perceptron_perception import costmap_eval as ke
from perceptron_perception.tracker import class_group

SEM_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'),
                      ('obj_idx', '<u4'), ('obj_tag', '<u4')])
KINDS = ('combined', 'obstacles', 'dynamic_risk', 'gt', 'sem')


class CostmapEvalNode(Node):

    def __init__(self):
        super().__init__('costmap_eval_node')
        self.declare_parameter('world_frame', 'carla_world')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('range', 20.0)
        self.declare_parameter('report_period', 15.0)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.world, self.robot_frame, self.range = p('world_frame'), p('robot_frame'), float(p('range'))
        self.scores = ke.CostmapScores()
        self.pending, self.ready, self.risk_hist, self.future = {}, [], {}, []
        self.prev_obst = None
        self.ego_id = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        for name in ('combined', 'obstacles', 'dynamic_risk'):
            self.create_subscription(OccupancyGrid, '/costmap/' + name, lambda m, k=name: self._add(k, m), 5)
        self.create_subscription(TrackedObjectArray, '/carla/ground_truth/objects', lambda m: self._add('gt', m), 10)
        self.create_subscription(PointCloud2, '/carla/ground_truth/lidar_semantic', lambda m: self._add('sem', m), qos)
        self.create_subscription(Float32MultiArray, '/costmap/stats', lambda m: self.scores.stats.append(list(m.data)), 10)
        self.create_subscription(String, '/carla/bridge_info', self._info_cb, latched)
        self.create_timer(0.05, self._drain)
        self.create_timer(float(p('report_period')), self._report)

    def _info_cb(self, msg):
        try:
            self.ego_id = int(json.loads(msg.data)['ego_id'])
        except (ValueError, KeyError):
            pass

    def _add(self, kind, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        e = self.pending.setdefault(key, {})
        e[kind] = msg
        if len(e) == len(KINDS):
            del self.pending[key]
            self.ready.append((key, e))
        if len(self.pending) > 60:
            for k in sorted(self.pending)[:-60]:
                del self.pending[k]

    def _drain(self):
        keep = []
        for key, e in self.ready:
            when = Time(seconds=key[0], nanoseconds=key[1])
            frame = e['combined'].header.frame_id
            if (self.tf_buffer.can_transform(frame, self.world, when) and
                    self.tf_buffer.can_transform(frame, self.robot_frame, when) and
                    self.tf_buffer.can_transform(frame, e['sem'].header.frame_id, when)):
                self._evaluate(key, e, frame, when)
            else:
                keep.append((key, e))
        self.ready = keep[-40:]

    def _evaluate(self, key, e, frame, when):
        t = key[0] + key[1] * 1e-9
        Mw = cp.transform_to_matrix(self.tf_buffer.lookup_transform(frame, self.world, when))
        robot = cp.transform_to_matrix(self.tf_buffer.lookup_transform(frame, self.robot_frame, when))[:2, 3]
        Ms = cp.transform_to_matrix(self.tf_buffer.lookup_transform(frame, e['sem'].header.frame_id, when))
        comb, origin, res = ke.grid_array(e['combined'])
        obst, o_origin, _ = ke.grid_array(e['obstacles'])
        risk, r_origin, _ = ke.grid_array(e['dynamic_risk'])
        self.scores.frames += 1

        sem = np.frombuffer(e['sem'].data, dtype=SEM_DTYPE, count=e['sem'].width * e['sem'].height)
        sxy = cp.transform_points(Ms, np.column_stack([sem['x'], sem['y'], sem['z']]).astype(float))[:, :2]
        hits = dict(zip(*np.unique(sem['obj_idx'], return_counts=True)))

        # truth in this map's frame
        yaw_off = math.atan2(Mw[1, 0], Mw[0, 0])
        actors = {}
        for o in e['gt'].objects:
            c = cp.transform_points(Mw, np.array([[o.pose.pose.position.x, o.pose.pose.position.y, 0.0]]))[0, :2]
            q = o.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)) + yaw_off
            actors[o.id] = {'xy': c, 'yaw': yaw, 'size': (o.size.x, o.size.y), 'group': class_group(o.class_id),
                            'speed': math.hypot(o.twist.twist.linear.x, o.twist.twist.linear.y)}

        # obstacle recall
        for aid, a in actors.items():
            if aid == self.ego_id or hits.get(aid, 0) < 3 or np.hypot(*(a['xy'] - robot)) > self.range:
                continue
            ii, jj = ke.cells_in_box(origin, res, comb.shape, a['xy'], a['yaw'], a['size'])
            covered = len(ii) > 0 and comb[ii, jj].max() >= 99
            self.scores.add_recall(a['group'], a['speed'] >= 0.5, covered)

        # false obstacles: lethal cells of the obstacle layer that only saw ground this frame
        i = ((sxy[:, 0] - o_origin[0]) / res).astype(int)
        j = ((sxy[:, 1] - o_origin[1]) / res).astype(int)
        ok = (i >= 0) & (i < obst.shape[0]) & (j >= 0) & (j < obst.shape[1])
        is_ground = np.isin(sem['obj_tag'], ce.GROUND_TAGS) & (sem['obj_idx'] != (self.ego_id or 0))
        not_ground = ~np.isin(sem['obj_tag'], ce.GROUND_TAGS) & (sem['obj_idx'] != (self.ego_id or 0))
        g_cells = set(zip(i[ok & is_ground], j[ok & is_ground]))
        o_cells = set(zip(i[ok & not_ground], j[ok & not_ground]))
        fp_cells = set()
        for c in g_cells | o_cells:
            if obst[c] >= 99:
                if c in o_cells:
                    self.scores.tp += 1
                else:
                    self.scores.fp += 1
                    fp_cells.add(c)
        if fp_cells:                       # which ground class those "only ground" cells saw
            sel = ok & is_ground
            for ci, cj, tag in zip(i[sel], j[sel], sem['obj_tag'][sel]):
                if (ci, cj) in fp_cells:
                    self.scores.fp_tags[int(tag)] = self.scores.fp_tags.get(int(tag), 0) + 1

        # dynamic coverage: keep this risk grid, schedule the check at t + 1 s, t + 2 s
        self.risk_hist[round(t, 3)] = (risk.copy(), r_origin, res)
        for k in [k for k in self.risk_hist if k < t - 2.5]:
            del self.risk_hist[k]
        for aid, a in actors.items():
            if a['speed'] >= 0.5 and a['group'] != 'other' and np.hypot(*(a['xy'] - robot)) <= self.range:
                for h in (1.0, 2.0):
                    self.future.append((round(t + h, 3), round(t, 3), aid, h))
        keep = []
        for target, t0, aid, h in self.future:
            if target > round(t, 3) + 1e-6:
                keep.append((target, t0, aid, h))
            elif abs(target - round(t, 3)) < 1e-6 and aid in actors and t0 in self.risk_hist:
                g, o, r = self.risk_hist[t0]
                v = ke.value_at(g, o, r, actors[aid]['xy'])
                if v is not None:
                    self.scores.dyn[h].append(v)
        self.future = keep

        cur = (obst, o_origin, res)
        if self.prev_obst is not None:
            self.scores.flicker.append(ke.flicker(self.prev_obst, cur))
        self.prev_obst = cur

    def _report(self):
        if not self.scores.frames:
            self.get_logger().warn('nothing scored yet: needs carla_bridge (--lidar-gt) and costmap_node',
                                   throttle_duration_sec=30.0)
            return
        self.get_logger().info('\n' + ke.format_summary(self.scores.summary(), 'costmap vs ground truth, since start'))


def main(args=None):
    rclpy.init(args=args)
    node = CostmapEvalNode()
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
