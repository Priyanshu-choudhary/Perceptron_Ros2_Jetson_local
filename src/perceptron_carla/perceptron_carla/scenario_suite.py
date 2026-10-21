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

Output: <out>/report.md (all scenarios), <out>/results.json, and per scenario
<out>/<name>/{metrics.json, bridge.log, stack.log, gt_*.csv, bag/}. Exit code 1 if any FAIL.
"""

import argparse
import json
import os
import signal
import subprocess
import time

import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection3DArray

from perceptron_carla import scenarios as sc
from perceptron_msgs.msg import PredictedObjectArray, TrackedObjectArray

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

    def _on_clock(self, msg):   # not _clock: rclpy.Node has its own
        self.sim = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.clock_wall = time.time()

    def _arrive(self, k):
        if self.sim is not None:
            self.arrivals[k].append(self.sim)

    def reset(self):
        self.sim = None
        for v in self.arrivals.values():
            v.clear()

    def spin_until(self, cond, wall_timeout):
        end = time.time() + wall_timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if cond():
                return True
        return False


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
        ['--no-semantic', '--log-dir', d]
    mon.reset()
    bridge, blog = start(bridge_cmd, os.path.join(d, 'bridge.log'))
    stack = slog = bag = baglog = None
    wall0 = time.time()
    try:
        if not mon.spin_until(lambda: mon.sim is not None or bridge.poll() is not None, 180.0) or mon.sim is None:
            notes.append('bridge did not start: ' + tail(os.path.join(d, 'bridge.log')).replace('\n', ' | '))
            return finish(name, spec, metrics, notes, d)
        sim0 = mon.sim
        stack, slog = start(['ros2', 'launch', 'perceptron_carla', 'scenario_stack.launch.py',
                             'stack:=' + spec['stack'], 'out_dir:=' + d], os.path.join(d, 'stack.log'))
        if args.bag != 'none':
            topics = sc.BAG_FULL if args.bag == 'full' else sc.BAG_LIGHT
            bag, baglog = start(['ros2', 'bag', 'record', '-o', os.path.join(d, 'bag')] + topics,
                                os.path.join(d, 'bag.log'))

        mon.spin_until(lambda: mon.sim >= sim0 + args.warmup, 120.0)
        t0, w0 = mon.sim, time.time()
        for v in mon.arrivals.values():
            v.clear()
        ok = mon.spin_until(lambda: mon.sim >= t0 + duration or bridge.poll() is not None
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
        out_topic = 'costmap' if spec['stack'] == 'perception' else 'odom_global'
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
