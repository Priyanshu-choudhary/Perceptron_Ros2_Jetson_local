#!/usr/bin/env python3
"""Run the Stage 9 scenario suite against a running CARLA and report PASS / FAIL per metric.

    # Windows: CarlaUE4.exe.  WSL (nothing else running):
    ros2 run perceptron_carla scenario_suite                      # all scenarios, ~15 min
    ros2 run perceptron_carla scenario_suite crossing city        # some of them
    ros2 run perceptron_carla scenario_suite --list
    ros2 run perceptron_carla scenario_suite --bag full           # + lidar / camera in the bags

For each scenario (definitions and limits in scenarios.py) it:
  1. starts carla_bridge with the scenario's seeded arguments (ground-truth CSV into the run dir)
  2. starts the stack and its evaluators (scenario_stack.launch.py), and a rosbag
  3. warms up, then scores `duration` sim seconds: topic rates, node health, and the
     evaluators' since-start summaries (each writes <run dir>/<evaluator>.json)
  4. stops everything in order (bag, stack, bridge) so files are complete
  5. checks the metrics against the limits

Navigation scenarios (stack 'navigation', Stage 11) start carla_nav.launch.py instead, send
the scenario's GPS goal once Nav2 is up, and score the drive from the ground truth: reached,
time, collisions (the bridge's collision sensor), minimum true gap between the car's body and
any pedestrian / vehicle box, distance driven, time stopped.

Output: <out>/report.md (all scenarios), <out>/results.json, and per scenario
<out>/<name>/{metrics.json, bridge.log, stack.log, gt_*.csv, bag/}. Exit code 1 if any FAIL.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import time
import zlib

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, NavSatFix, PointCloud2
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from perceptron_carla import scenarios as sc
from perceptron_msgs.msg import PredictedObjectArray, TrackedObjectArray
from perceptron_navigation import geo
from perceptron_perception.tracker import class_group

TYPES = {'lidar': PointCloud2, 'camera': Image, 'obstacles': PointCloud2, 'objects': Detection3DArray,
         'tracks': TrackedObjectArray, 'predictions': PredictedObjectArray, 'costmap': OccupancyGrid,
         'odom_local': Odometry, 'odom_global': Odometry}


class Monitor(Node):
    """Sim clock and message arrival (in sim time) on the rate topics."""

    def __init__(self):
        super().__init__('scenario_suite')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.sim = None
        self.clock_wall = 0.0
        self.arrivals = {k: [] for k in sc.RATE_TOPICS}
        self.create_subscription(Clock, "/clock", self._on_clock, qos)
        for k, topic in sc.RATE_TOPICS.items():
            self.create_subscription(TYPES[k], topic, lambda m, k=k: self._arrive(k), qos)
        # navigation scoring
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.info = self.graph = self.ego = self.nav = None
        self.create_subscription(String, '/carla/bridge_info', lambda m: setattr(self, 'info', json.loads(m.data)),
                                 latched)
        self.create_subscription(String, '/carla/map/road_graph',
                                 lambda m: setattr(self, 'graph', json.loads(m.data)), latched)
        self.create_subscription(String, '/gps_goal/status', self._nav_status, 10)
        self.create_subscription(String, '/carla/ego/collision', self._collision, 10)
        self.create_subscription(Odometry, '/carla/ground_truth/ego_odom', self._ego, 10)
        self.create_subscription(TrackedObjectArray, '/carla/ground_truth/objects', self._objects, 10)
        self.pub_goal = self.create_publisher(NavSatFix, '/gps_goal', 10)
        self.pub_rel = self.create_publisher(PointStamped, '/gps_goal/relative', 10)

    def _on_clock(self, msg):   # not _clock: rclpy.Node has its own
        self.sim = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.clock_wall = time.time()

    def _arrive(self, k):
        if self.sim is not None:
            self.arrivals[k].append(self.sim)

    def reset(self):
        self.sim = None
        self.info = self.graph = self.ego = self.nav = None
        for v in self.arrivals.values():
            v.clear()

    # ---------------------------------------------------------------- navigation
    def _nav_status(self, msg):
        if self.nav is None or self.nav['done']:
            return
        st = json.loads(msg.data)
        self.nav['status'] = st
        if st['state'] == 'driving':
            self.nav['driving'] = True
            if st.get('route_length'):
                self.nav['route_m'] = st['route_length']
        elif self.nav['driving'] and st['state'] != 'idle':
            self.nav['done'], self.nav['t_end'] = True, self.sim

    def _collision(self, msg):
        if self.nav is not None and not self.nav['done']:
            self.nav['collisions'].append(json.loads(msg.data))

    def _ego(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        prev, self.ego = self.ego, (p.x, p.y, yaw, msg.twist.twist.linear.x, t)
        n = self.nav
        if n is not None and n['driving'] and not n['done'] and prev is not None:
            n['driven_m'] += math.hypot(p.x - prev[0], p.y - prev[1])
            if abs(msg.twist.twist.linear.x) < 0.1:
                n['stopped_s'] += max(t - prev[4], 0.0)

    def _objects(self, msg):
        n = self.nav
        if n is None or n['done'] or self.ego is None or self.info is None:
            return
        ex, ey, eyaw = self.ego[:3]
        x0, x1, y0, y1 = self.info['host_box_base_footprint'][:4]
        ego = sc.box_corners(ex, ey, eyaw, x0, x1, y0, y1)
        for o in msg.objects:
            if o.id == self.info.get('ego_id'):
                continue
            ox, oy = o.pose.pose.position.x, o.pose.pose.position.y
            if math.hypot(ox - ex, oy - ey) > 20.0:
                continue
            q = o.pose.pose.orientation
            oyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            box = sc.box_corners(ox, oy, oyaw, -o.size.x / 2, o.size.x / 2, -o.size.y / 2, o.size.y / 2)
            g = class_group(o.class_id)
            gap = sc.rect_distance(ego, box)
            n['clearance'][g] = min(n['clearance'].get(g, 20.0), gap)
            near = gap < 0.5
            if near and not n['near'].get(o.id):                 # a new near pass by this actor
                n['near_misses'].append({'t': round(self.sim, 1), 'id': o.id, 'group': g, 'gap': round(gap, 2),
                                         'ego_v': round(self.ego[3], 2)})
            n['near'][o.id] = near

    def pick_goal(self, spec, seed):
        """(lat, lon) for a 'road' goal, or None; see scenarios.SCENARIOS."""
        lo, hi = spec['road']
        g = geo.RoadGraph(self.graph['lanes'])
        ex, ey, eyaw = self.ego[:3]
        from scipy.sparse.csgraph import dijkstra
        s, _ = g.nearest((ex, ey))
        dist = dijkstra(g.adj, directed=False, indices=s)
        ok = (dist >= lo) & (dist <= hi)
        if spec.get('forward'):
            rel = g.p - (ex, ey)
            ok &= (rel @ (math.cos(eyaw), math.sin(eyaw))) > np.linalg.norm(rel, axis=1) * 0.5
        cand = np.flatnonzero(ok)
        if not len(cand):
            return None
        k = int(np.random.default_rng(seed).choice(cand))
        lat0, lon0 = self.info['gnss_origin']
        return geo.enu_to_latlon(g.p[k][0], g.p[k][1], lat0, lon0) + (float(dist[k]),)

    def send_goal(self, spec, seed, notes):
        """Wait for Nav2 + gps_goal_node, send the scenario goal, re-send until it is driving."""
        need_graph = 'road' in spec['goal']
        ready = lambda: (self.info is not None and self.ego is not None and (self.graph or not need_graph)  # noqa: E731
                         and self.pub_goal.get_subscription_count() > 0
                         and 'bt_navigator' in self.get_node_names())
        if not self.spin_until(ready, 90.0):
            notes.append('navigation stack or bridge outputs not ready: goal not sent')
            return False
        self.spin_until(lambda: False, 4.0)            # lifecycle activation
        goal = spec['goal']
        if 'road' in goal:
            pick = self.pick_goal(goal, seed)
            if pick is None:
                notes.append('no road point %s m away' % goal['road'])
                return False
            lat, lon, road_m = pick
            msg, pub = NavSatFix(), self.pub_goal
            msg.latitude, msg.longitude = lat, lon
            notes.append('goal GPS %.7f, %.7f (%.0f m by road)' % (lat, lon, road_m))
        else:
            msg, pub = PointStamped(), self.pub_rel
            msg.header.frame_id = 'base_footprint'
            msg.point.x = float(goal['ahead'])
            notes.append('goal %.0f m straight ahead' % goal['ahead'])
        self.nav = {'t_send': self.sim, 't_end': None, 'driving': False, 'done': False, 'status': None,
                    'collisions': [], 'driven_m': 0.0, 'stopped_s': 0.0, 'clearance': {}, 'route_m': None,
                    'near': {}, 'near_misses': []}
        for _ in range(6):
            pub.publish(msg)
            if self.spin_until(lambda: self.nav['driving'], 10.0):
                return True
        notes.append('Nav2 never started driving: ' + json.dumps(self.nav['status']))
        return False

    def spin_until(self, cond, wall_timeout):
        end = time.time() + wall_timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if cond():
                return True
        return False


def semantic_args(spec):
    """Navigation runs keep the segmentation camera (vision road layer), except with --crossing:
    CARLA 0.9.15 froze ~20 s into every crossing run with the segmentation camera on."""
    if spec['stack'] == 'navigation' and '--crossing' not in spec['bridge']:
        return []
    return ['--no-semantic']


def start(cmd, log_path):
    log = open(log_path, 'w')
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def stop(proc, log, wait=25.0, name=''):
    """SIGINT the process group (ros2 launch / ros2 run pass it on), escalate if needed."""
    if proc is None:
        return
    for sig, t in ((signal.SIGINT, wait), (signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        if proc.poll() is not None:
            break
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=t)
        except subprocess.TimeoutExpired:
            print('  %s did not stop on %s' % (name, signal.Signals(sig).name))
    log.close()


def tail(path, n=8):
    try:
        with open(path) as f:
            return ''.join(f.readlines()[-n:])
    except OSError:
        return ''


def run_scenario(mon, name, out, args):
    spec = sc.SCENARIOS[name]
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    duration = spec['duration'] * args.duration_scale
    notes, metrics = [], {}
    print('\n=== %s: %s\n    bridge %s, scoring %.0f sim s' % (name, spec['about'], spec['bridge'], duration))

    bridge_cmd = ['ros2', 'run', 'perceptron_carla', 'carla_bridge'] + spec['bridge'].split() + \
        semantic_args(spec) + ['--log-dir', d]
    mon.reset()
    bridge, blog = start(bridge_cmd, os.path.join(d, 'bridge.log'))
    stack = slog = bag = baglog = None
    wall0 = time.time()
    try:
        if not mon.spin_until(lambda: mon.sim is not None or bridge.poll() is not None, 180.0) or mon.sim is None:
            notes.append('bridge did not start: ' + tail(os.path.join(d, 'bridge.log')).replace('\n', ' | '))
            return finish(name, spec, metrics, notes, d)
        sim0 = mon.sim
        if spec['stack'] == 'navigation':
            stack_cmd = ['ros2', 'launch', 'perceptron_carla', 'carla_nav.launch.py']
        else:
            stack_cmd = ['ros2', 'launch', 'perceptron_carla', 'scenario_stack.launch.py',
                         'stack:=' + spec['stack'], 'out_dir:=' + d]
        stack, slog = start(stack_cmd, os.path.join(d, 'stack.log'))
        if args.bag != 'none':
            topics = sc.BAG_FULL if args.bag == 'full' else sc.BAG_LIGHT
            bag, baglog = start(['ros2', 'bag', 'record', '-o', os.path.join(d, 'bag')] + topics,
                                os.path.join(d, 'bag.log'))

        mon.spin_until(lambda: mon.sim >= sim0 + args.warmup, 120.0)
        nav = spec['stack'] == 'navigation'
        if nav and not mon.send_goal(spec, zlib.crc32(name.encode()) % 1000, notes):   # stable across runs
            mon.nav = None
        t0, w0 = mon.sim, time.time()
        for v in mon.arrivals.values():
            v.clear()
        ok = mon.spin_until(lambda: mon.sim >= t0 + duration or bridge.poll() is not None
                            or (nav and (mon.nav is None or mon.nav['done']))
                            or time.time() - mon.clock_wall > 30.0, duration * 15 + 120)
        t1, w1 = mon.sim, time.time()
        if bridge.poll() is not None:
            notes.append('bridge exited early: ' + tail(os.path.join(d, 'bridge.log')).replace('\n', ' | '))
        elif time.time() - mon.clock_wall > 30.0 or not ok:
            notes.append('the simulation stalled (no /clock for 30 s): CARLA frozen?')
        span = max(t1 - t0, 1e-6)
        for k, ts in mon.arrivals.items():
            if ts:
                metrics['rate.%s_hz' % k] = len(ts) / span
        metrics['rate.realtime_factor'] = span / max(w1 - w0, 1e-6)
        metrics['sim_seconds_scored'] = span
        out_topic = 'odom_global' if spec['stack'] == 'localization' else 'costmap'
        if nav and mon.nav is not None:
            n = mon.nav
            state = (n['status'] or {}).get('state')
            metrics['nav.reached'] = int(state == 'reached')
            metrics['nav.time_s'] = (n['t_end'] or mon.sim) - n['t_send']
            metrics['nav.collisions'] = len(n['collisions'])
            metrics['nav.driven_m'] = n['driven_m']
            metrics['nav.stopped_s'] = n['stopped_s']
            metrics['nav.mean_speed'] = n['driven_m'] / max(metrics['nav.time_s'], 1e-6)
            if n['route_m']:
                metrics['nav.route_m'] = n['route_m']
            left = (n['status'] or {}).get('distance_left')
            if left is not None:
                metrics['nav.distance_left'] = left
            for g, gap in n['clearance'].items():
                metrics['nav.min_clearance_%s' % g] = gap
            for g in ('pedestrian', 'vehicle'):
                metrics.setdefault('nav.min_clearance_%s' % g, 20.0)     # none came within 20 m
                metrics['nav.near_misses_%s' % g] = sum(1 for m in n['near_misses'] if m['group'] == g)
            for m in n['near_misses']:
                notes.append('near pass at %.1f s: %s %d within %.2f m, ego %.2f m/s'
                             % (m['t'], m['group'], m['id'], m['gap'], m['ego_v']))
            if not n['done']:
                notes.append('goal not reached in %.0f sim s (state %s)' % (duration, state))
            for c in n['collisions']:
                notes.append('collision at %.1f s with %s (impulse %s)' % (c['t'], c['other_type'], c['impulse']))
        metrics['health.outputs_after_dropout'] = int(any(t >= t1 - 10.0 for t in mon.arrivals[out_topic]))

        alive = set(mon.get_node_names())
        missing = [n for n in sc.STACK_NODES[spec['stack']] if n not in alive]
        metrics['health.nodes_missing'] = len(missing)
        if missing:
            notes.append('nodes not running at the end: ' + ', '.join(missing))
    finally:
        stop(bag, baglog, 15.0, 'rosbag') if bag else None
        stop(stack, slog, 25.0, 'stack') if stack else None
        stop(bridge, blog, 40.0, 'bridge')
    print('    ran %.0f s wall' % (time.time() - wall0))
    return finish(name, spec, metrics, notes, d)


def finish(name, spec, metrics, notes, d):
    summaries = {}
    for ev in sc.EVALUATORS[spec['stack']]:
        path = os.path.join(d, ev + '.json')
        try:
            with open(path) as f:
                summaries[ev] = json.load(f)['summary']
        except (OSError, ValueError, KeyError):
            notes.append('%s wrote no summary' % ev)
    metrics.update(sc.extract_metrics(summaries))
    rows = sc.check(metrics, sc.limits_for(name, metrics))
    result = {'name': name, 'about': spec['about'], 'bridge': spec['bridge'], 'verdict': sc.verdict(rows),
              'rows': rows, 'metrics': metrics, 'notes': notes}
    with open(os.path.join(d, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=1)
    print('    %s  (%s)' % (result['verdict'], ', '.join('%s %s' % (r[0], r[4]) for r in rows if r[4] != 'PASS')
                            or 'all limits met'))
    for n in notes:
        print('    note: ' + n)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description='Stage 9 CARLA scenario suite')
    p.add_argument('scenarios', nargs='*', help='names (default: all); see --list')
    p.add_argument('--list', action='store_true')
    p.add_argument('--out', default='', help='run directory (default ~/carla_runs/suite_<time>)')
    p.add_argument('--bag', choices=('light', 'full', 'none'), default='light',
                   help='light: poses, ground truth, perception outputs (~1 MB/s); full: + lidar and camera')
    p.add_argument('--warmup', type=float, default=8.0, help='sim s before scoring starts')
    p.add_argument('--duration-scale', type=float, default=1.0, help='scale every scenario duration')
    args, _ = p.parse_known_args(argv)
    if args.list:
        for n, s in sc.SCENARIOS.items():
            print('%-18s %5.0f s  %s\n%18s          bridge: %s' % (n, s['duration'], s['about'], '', s['bridge']))
        return 0
    names = args.scenarios or list(sc.SCENARIOS)
    unknown = [n for n in names if n not in sc.SCENARIOS]
    if unknown:
        p.error('unknown scenario(s) %s; known: %s' % (unknown, ', '.join(sc.SCENARIOS)))
    out = os.path.expanduser(args.out or time.strftime('~/carla_runs/suite_%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)

    rclpy.init()
    mon = Monitor()
    results = []
    try:
        for n in names:
            results.append(run_scenario(mon, n, out, args))
    except KeyboardInterrupt:
        print('\ninterrupted; reporting what ran')
    finally:
        mon.destroy_node()
        rclpy.try_shutdown()
    report = sc.format_report(results, 'CARLA scenario suite, %s' % time.strftime('%Y-%m-%d %H:%M'))
    with open(os.path.join(out, 'report.md'), 'w') as f:
        f.write(report)
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print('\n' + report)
    print('report: %s' % os.path.join(out, 'report.md'))
    return 1 if any(r['verdict'] == 'FAIL' for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
