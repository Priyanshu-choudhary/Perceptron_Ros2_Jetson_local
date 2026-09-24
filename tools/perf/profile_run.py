#!/usr/bin/env python3
"""Launch one configuration of the nav stack, measure it, tear it down.

Phases
  startup   ros2 launch ... ; wait until both lifecycle managers report active
  localize  publish /initialpose, settle
  idle      localized, no goal: measure per-process CPU / memory
  nav       /goal_pose sent, controller running: measure again

CPU is read from /proc/<pid>/stat (utime+stime) for every process in the
launch's process tree, so the measurement is exact CPU time, not a sampled
percentage. 1.00 core = one logical CPU of this machine fully busy.
Memory is PSS from /proc/<pid>/smaps_rollup (shared libraries split fairly
between the processes that map them, so the column sums honestly) plus RSS.

This process (rclpy helper node + sampler) is NOT in the launch tree and is
excluded from every number.
"""

import argparse
import glob
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
CLK = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')


# --------------------------------------------------------------- /proc reading
def read_stat(pid):
    with open(f'/proc/{pid}/stat') as f:
        s = f.read()
    rp = s.rfind(')')
    fields = s[rp + 2:].split()
    # fields[0] = state (field 3); utime = field 14 -> idx 11; stime 15 -> idx 12
    return int(fields[11]) + int(fields[12]), int(fields[17])  # ticks, num_threads(20)->idx17


def read_threads(pid):
    out = {}
    for tdir in glob.glob(f'/proc/{pid}/task/*'):
        try:
            tid = int(os.path.basename(tdir))
            with open(tdir + '/stat') as f:
                s = f.read()
            lp, rp = s.find('('), s.rfind(')')
            comm = s[lp + 1:rp]
            fields = s[rp + 2:].split()
            out[tid] = (comm, int(fields[11]) + int(fields[12]))
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass
    return out


def read_mem(pid):
    d = {}
    try:
        with open(f'/proc/{pid}/smaps_rollup') as f:
            for line in f:
                parts = line.split()
                if parts[0] in ('Rss:', 'Pss:', 'Private_Clean:', 'Private_Dirty:', 'Swap:'):
                    d[parts[0][:-1]] = int(parts[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    d['USS'] = d.get('Private_Clean', 0) + d.get('Private_Dirty', 0)
    return d  # kB


def read_sys():
    with open('/proc/stat') as f:
        vals = [int(x) for x in f.readline().split()[1:]]
    idle = vals[3] + vals[4]
    return sum(vals), idle


def label_for(p):
    try:
        cmd = p.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    joined = ' '.join(cmd)
    m = re.search(r'__node:=([\w/]+)', joined)
    if m:
        return m.group(1)
    if 'ros2' in joined and ' launch ' in joined:
        return 'ros2_launch'
    if cmd:
        return os.path.basename(cmd[1] if cmd[0].endswith('python3') and len(cmd) > 1 else cmd[0])
    return p.name()


class Tree:
    """The launch process and every descendant, relabelled on each scan."""

    def __init__(self, root_pid):
        self.root = psutil.Process(root_pid)
        self.labels = {}

    def pids(self):
        try:
            procs = [self.root] + self.root.children(recursive=True)
        except psutil.NoSuchProcess:
            return {}
        out = {}
        for p in procs:
            if p.pid not in self.labels:
                lab = label_for(p)
                if lab is None:
                    continue
                self.labels[p.pid] = lab
            out[p.pid] = self.labels[p.pid]
        return out


EXTRA_PIDS = {}


def measure(tree, seconds, mem_every=5):
    """Exact CPU per process over the window, 1 s series, thread split, PSS."""
    pids = tree.pids()
    t_wall0 = time.time()
    ex0 = {}
    for lab, pid in EXTRA_PIDS.items():
        try:
            ex0[lab] = read_stat(pid)[0]
        except (FileNotFoundError, ProcessLookupError):
            pass
    t0 = time.monotonic()
    sys0 = read_sys()
    start = {}
    th_start = {}
    for pid in pids:
        try:
            start[pid] = read_stat(pid)[0]
            th_start[pid] = read_threads(pid)
        except (FileNotFoundError, ProcessLookupError):
            pass
    series = {pid: [] for pid in pids}
    prev = dict(start)
    prev_t = t0
    mem = {pid: [] for pid in pids}
    k = 0
    while time.monotonic() - t0 < seconds:
        time.sleep(1.0)
        now = time.monotonic()
        k += 1
        for pid in list(prev):
            try:
                ticks = read_stat(pid)[0]
            except (FileNotFoundError, ProcessLookupError):
                continue
            series[pid].append((ticks - prev[pid]) / CLK / (now - prev_t))
            prev[pid] = ticks
            if k % mem_every == 1:
                m = read_mem(pid)
                if m:
                    mem[pid].append(m)
        prev_t = now
    t1 = time.monotonic()
    sys1 = read_sys()
    dt = t1 - t0
    res = {}
    for pid, lab in pids.items():
        if pid not in start:
            continue
        try:
            end = read_stat(pid)
            th_end = read_threads(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        cores = (end[0] - start[pid]) / CLK / dt
        ser = series.get(pid) or [0.0]
        threads = {}
        for tid, (comm, ticks) in th_end.items():
            base = th_start[pid].get(tid, (comm, ticks if tid not in th_start[pid] else 0))[1]
            c = (ticks - base) / CLK / dt
            threads.setdefault(comm, 0.0)
            threads[comm] += c
        ms = mem.get(pid) or []
        key = lab
        n = 2
        while key in res:
            key = f'{lab}#{n}'
            n += 1
        res[key] = {
            'pid': pid,
            'cores': cores,
            'p95': sorted(ser)[int(0.95 * (len(ser) - 1))],
            'max': max(ser),
            'stdev': statistics.pstdev(ser) if len(ser) > 1 else 0.0,
            'threads_total': end[1],
            'threads_cpu': dict(sorted(threads.items(), key=lambda kv: -kv[1])[:6]),
            'pss_kb': statistics.mean(m['Pss'] for m in ms) if ms else None,
            'rss_kb': statistics.mean(m['Rss'] for m in ms) if ms else None,
            'uss_kb': statistics.mean(m['USS'] for m in ms) if ms else None,
        }
    extra = {}
    for lab, pid in EXTRA_PIDS.items():
        try:
            extra[lab] = {'cores': (read_stat(pid)[0] - ex0[lab]) / CLK / dt,
                          'rss_kb': (read_mem(pid) or {}).get('Rss')}
        except (KeyError, FileNotFoundError, ProcessLookupError):
            pass
    tot0, idle0 = sys0
    tot1, idle1 = sys1
    ncpu = os.cpu_count()
    sys_busy = (1.0 - (idle1 - idle0) / max(1, tot1 - tot0)) * ncpu
    return {
        'seconds': dt,
        'procs': res,
        'total_cores': sum(v['cores'] for v in res.values()),
        'total_pss_mb': sum((v['pss_kb'] or 0) for v in res.values()) / 1024.0,
        'total_rss_mb': sum((v['rss_kb'] or 0) for v in res.values()) / 1024.0,
        'system_busy_cores': sys_busy,
        'ncpu': ncpu,
        'extra': extra,
        't_start': t_wall0,
        't_end': time.time(),
    }


# ------------------------------------------------------------------ ROS helper
class Helper:
    """rclpy node living in THIS process: pose, goal, fake-odom, counters."""

    def __init__(self, odom_topic):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Twist
        from nav_msgs.msg import Odometry
        from std_srvs.srv import Empty
        self.rclpy = rclpy
        rclpy.init()
        self.node = Node('profiler_helper')
        self.PoseWithCovarianceStamped = PoseWithCovarianceStamped
        self.PoseStamped = PoseStamped
        self.Odometry = Odometry
        self.init_pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.odom_topic = odom_topic
        self.odom_pub = self.node.create_publisher(Odometry, odom_topic, 10) if odom_topic else None
        self.counts = {'cmd_vel_nav': 0, 'cmd_vel': 0}
        self.last_cmd = Twist()
        self.nonzero_cmd = 0
        self.node.create_subscription(Twist, '/cmd_vel_nav', self._cmd_nav, 10)
        self.node.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.nomotion = self.node.create_client(Empty, '/request_nomotion_update')
        self.Empty = Empty
        self.fake_odom_on = False
        self.nomotion_hz = 0.0
        self._stop = False
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()
        self.node.create_timer(1.0 / 92.0, self._odom_tick)
        self._last_nomotion = 0.0
        self.nomotion_calls = 0

    def _spin(self):
        while not self._stop and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.nomotion_hz > 0:
                now = time.monotonic()
                if now - self._last_nomotion >= 1.0 / self.nomotion_hz:
                    self._last_nomotion = now
                    if self.nomotion.service_is_ready():
                        self.nomotion.call_async(self.Empty.Request())
                        self.nomotion_calls += 1

    def _cmd_nav(self, msg):
        self.counts['cmd_vel_nav'] += 1

    def _cmd(self, msg):
        self.counts['cmd_vel'] += 1
        self.last_cmd = msg
        if abs(msg.linear.x) > 1e-3 or abs(msg.angular.z) > 1e-3:
            self.nonzero_cmd += 1

    def _odom_tick(self):
        # Velocity loop-back: controller_server's odom feed reports the speed
        # Nav2 itself last commanded, as a robot tracking cmd_vel would. Only
        # controller_server listens to this topic (see params override).
        if not self.odom_pub:
            return
        m = self.Odometry()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.child_frame_id = 'base_footprint'
        if self.fake_odom_on:
            m.twist.twist.linear.x = self.last_cmd.linear.x
            m.twist.twist.angular.z = self.last_cmd.angular.z
        self.odom_pub.publish(m)

    def set_pose(self, x, y, yaw, n=3):
        for _ in range(n):
            m = self.PoseWithCovarianceStamped()
            m.header.frame_id = 'map'
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.pose.pose.position.x = x
            m.pose.pose.position.y = y
            m.pose.pose.orientation.z = math.sin(yaw / 2)
            m.pose.pose.orientation.w = math.cos(yaw / 2)
            cov = [0.0] * 36
            cov[0] = cov[7] = 0.05 ** 2
            cov[35] = 0.03 ** 2
            m.pose.covariance = cov
            self.init_pub.publish(m)
            time.sleep(0.5)

    def send_goal(self, x, y, yaw, latest=False):
        m = self.PoseStamped()
        m.header.frame_id = 'map'
        if not latest:  # stamp 0 = "use the latest transform" (robust on a slow TF tree)
            m.header.stamp = self.node.get_clock().now().to_msg()
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.orientation.z = math.sin(yaw / 2)
        m.pose.orientation.w = math.cos(yaw / 2)
        self.goal_pub.publish(m)

    def set_params(self, node_name, params):
        from rcl_interfaces.srv import SetParameters
        from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
        cli = self.node.create_client(SetParameters, f'/{node_name}/set_parameters')
        if not cli.wait_for_service(timeout_sec=10.0):
            return 'no service'
        req = SetParameters.Request()
        for k, v in params.items():
            pv = ParameterValue()
            if isinstance(v, bool):
                pv.type, pv.bool_value = ParameterType.PARAMETER_BOOL, v
            elif isinstance(v, int):
                pv.type, pv.integer_value = ParameterType.PARAMETER_INTEGER, v
            elif isinstance(v, float):
                pv.type, pv.double_value = ParameterType.PARAMETER_DOUBLE, v
            else:
                pv.type, pv.string_value = ParameterType.PARAMETER_STRING, str(v)
            req.parameters.append(Parameter(name=k, value=pv))
        fut = cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 10:
            time.sleep(0.05)
        if not fut.done():
            return 'timeout'
        return [(r.successful, r.reason) for r in fut.result().results]

    def shutdown(self):
        self._stop = True
        time.sleep(0.2)
        try:
            self.node.destroy_node()
            self.rclpy.shutdown()
        except Exception:
            pass


class ZmqRate:
    """Counts what the Jetson actually sends (odom, scan) so every window
    records its input load. Publisher-side filtering: only these two topics
    are sent to this socket."""

    def __init__(self, url='tcp://192.168.1.7:5555'):
        import zmq
        self.zmq = zmq
        self.counts = {'odom': 0, 'scan': 0}
        self._stop = False
        ctx = zmq.Context.instance()
        self.s = ctx.socket(zmq.SUB)
        for t in (b'odom', b'scan'):
            self.s.setsockopt(zmq.SUBSCRIBE, t)
        self.s.setsockopt(zmq.RCVTIMEO, 200)
        self.s.connect(url)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            try:
                f = self.s.recv_multipart()
                k = f[0].decode()
                if k in self.counts:
                    self.counts[k] += 1
            except self.zmq.Again:
                pass

    def snap(self):
        return dict(self.counts)


# ------------------------------------------------------------------ lifecycle
KILL_PATTERNS = ['profile_nav.launch.py', 'nav.launch.py', 'jetson_bridge_node',
                 'robot_state_publisher', 'joint_state_publisher', 'battery_node',
                 'ekf_node', 'map_server', 'amcl', 'lifecycle_manager',
                 'aruco_localizer_node', 'rviz2', 'controller_server', 'planner_server',
                 'smoother_server', 'behavior_server', 'bt_navigator',
                 'waypoint_follower', 'velocity_smoother', 'path_overlay_node',
                 'component_container']


def kill_leftovers():
    me = os.getpid()
    for p in psutil.process_iter(['pid', 'cmdline']):
        if p.info['pid'] == me:
            continue
        cmd = ' '.join(p.info['cmdline'] or [])
        if 'jetson_robot_bridge.py' in cmd:   # the Jetson's own serial bridge: never touch
            continue
        if '/opt/ros/humble' in cmd or 'perceptron_test_ws' in cmd or 'ros2' in cmd or '--ros-args' in cmd:
            if any(k in cmd for k in KILL_PATTERNS) and 'profile_run.py' not in cmd:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
    time.sleep(1.0)


def wait_active(logpath, want, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            txt = open(logpath, errors='replace').read()
        except FileNotFoundError:
            txt = ''
        got = [w for w in want if re.search(w + r'.*Managed nodes are active', txt)]
        if len(got) == len(want):
            return time.time() - t0
        time.sleep(0.5)
    return None


def log_counts(txt):
    pats = {
        'missed_control_rate': r'Control loop missed its desired rate',
        'costmap_missed_rate': r'Map update loop missed its desired rate',
        'no_valid_traj': r'No valid trajector',
        'failed_progress': r'Failed to make progress',
        'rotating_shim': r'[Rr]otat(ing|e) to',
        'planning_failed': r'(Planning algorithm .* failed|failed to create plan|Failed to create a plan|GridBased plugin failed|Smac2D.* failed)',
        'tf_errors': r'(extrapolation|Lookup would require|TF_OLD_DATA|Could not transform|Timed out waiting for transform)',
        'bt_ack_timeout': r'Timed out while waiting for action server',
        'goal_succeeded': r'Goal succeeded',
        'goal_aborted': r'(Goal failed|aborted)',
        'warn_lines': r'\[WARN\]',
        'error_lines': r'\[ERROR\]',
    }
    return {k: len(re.findall(v, txt)) for k, v in pats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--launch-arg', action='append', default=[])
    ap.add_argument('--params-file', default='')
    ap.add_argument('--amcl-set', default='{}', help='JSON of runtime AMCL params')
    ap.add_argument('--idle', type=float, default=45)
    ap.add_argument('--nav', type=float, default=30)
    ap.add_argument('--settle', type=float, default=12)
    ap.add_argument('--pose', default='', help='x,y,yaw in map')
    ap.add_argument('--goal', default='', help='x,y,yaw in map')
    ap.add_argument('--nomotion-hz', type=float, default=0.0,
                    help='AMCL filter updates per second during nav (emulated motion)')
    ap.add_argument('--fake-odom-topic', default='/profiler/controller_odom')
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'results'))
    ap.add_argument('--keep-running', action='store_true')
    ap.add_argument('--launch-file', default=os.path.join(HERE, 'profile_nav.launch.py'))
    ap.add_argument('--zmq-url', default='tcp://192.168.1.7:5555')
    ap.add_argument('--goal-latest', action='store_true')
    ap.add_argument('--extra-pattern', action='append', default=[],
                    help='label=substring: also measure a process outside the launch tree')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    EXTRA_PIDS['profiler_driver(self)'] = os.getpid()
    for spec in a.extra_pattern:
        lab, pat = spec.split('=', 1)
        for p in psutil.process_iter(['pid', 'cmdline']):
            if pat in ' '.join(p.info['cmdline'] or []) and p.info['pid'] != os.getpid():
                EXTRA_PIDS[lab] = p.info['pid']
                break
    logpath = os.path.join(a.out, a.name + '.log')
    kill_leftovers()

    cmd = ['ros2', 'launch', a.launch_file] + a.launch_arg
    if a.params_file:
        cmd.append(f'params_file:={a.params_file}')
    print('[profiler]', ' '.join(cmd), flush=True)
    logf = open(logpath, 'w')
    t_launch = time.time()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    result = {'name': a.name, 'cmd': cmd, 'launch_args': a.launch_arg,
              'params_file': a.params_file, 'amcl_set': json.loads(a.amcl_set),
              'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    helper = None
    zr = ZmqRate(a.zmq_url)
    try:
        # AMCL must be seeded before Nav2 can activate: the global costmap
        # blocks in on_configure until map -> base_link exists.
        t_loc = wait_active(logpath, ['lifecycle_manager_localization'], 120)
        if t_loc is None:
            result['error'] = 'localization lifecycle never became active'
            raise RuntimeError(result['error'])
        tree = Tree(proc.pid)
        helper = Helper(a.fake_odom_topic)
        time.sleep(2.0)
        if a.amcl_set and json.loads(a.amcl_set):
            result['amcl_set_result'] = helper.set_params('amcl', json.loads(a.amcl_set))
            time.sleep(1.0)
        x, y, yaw = map(float, a.pose.split(','))
        t_act = None
        t_wait0 = time.time()
        while time.time() - t_wait0 < 150:
            # one pose, then give AMCL time to push a scan through its TF
            # filter; re-sending every 2 s livelocks AMCL on a slow CPU
            helper.set_pose(x, y, yaw, n=1)
            t_act = wait_active(logpath, ['lifecycle_manager_navigation'], 20.0)
            if t_act is not None:
                break
        result['startup_s'] = time.time() - t_launch if t_act is not None else None
        if t_act is None:
            result['error'] = 'navigation lifecycle never became active'
            raise RuntimeError(result['error'])
        t_act = result['startup_s']
        helper.set_pose(x, y, yaw)
        time.sleep(a.settle)
        # startup peak memory is not interesting; steady state is
        print(f'[profiler] {a.name}: active after {t_act:.1f}s, measuring idle {a.idle}s', flush=True)
        c0 = dict(helper.counts)
        z0 = zr.snap()
        result['idle'] = measure(tree, a.idle)
        result['idle']['msgs'] = {k: helper.counts[k] - c0[k] for k in c0}
        result['idle']['jetson_hz'] = {k: (zr.snap()[k] - z0[k]) / result['idle']['seconds'] for k in z0}

        if a.nav > 0 and a.goal:
            gx, gy, gyaw = map(float, a.goal.split(','))
            helper.fake_odom_on = True
            helper.nomotion_hz = a.nomotion_hz
            log_before = open(logpath, errors='replace').read()
            helper.send_goal(gx, gy, gyaw, latest=a.goal_latest)
            time.sleep(4.0)
            print(f'[profiler] {a.name}: goal sent, measuring nav {a.nav}s', flush=True)
            c0 = dict(helper.counts)
            nz0 = helper.nonzero_cmd
            nm0 = helper.nomotion_calls
            z0 = zr.snap()
            result['nav'] = measure(tree, a.nav)
            result['nav']['jetson_hz'] = {k: (zr.snap()[k] - z0[k]) / result['nav']['seconds'] for k in z0}
            result['nav']['msgs'] = {k: helper.counts[k] - c0[k] for k in c0}
            result['nav']['nonzero_cmd'] = helper.nonzero_cmd - nz0
            result['nav']['amcl_updates_requested'] = helper.nomotion_calls - nm0
            result['nav']['last_cmd'] = [helper.last_cmd.linear.x, helper.last_cmd.angular.z]
            log_after = open(logpath, errors='replace').read()
            result['nav']['log'] = log_counts(log_after[len(log_before):])
            helper.fake_odom_on = False
            helper.nomotion_hz = 0.0
        result['log_total'] = log_counts(open(logpath, errors='replace').read())
        if a.keep_running:
            print('[profiler] keep-running: press Ctrl+C', flush=True)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa
        result.setdefault('error', repr(e))
        print('[profiler] ERROR', e, flush=True)
    finally:
        if helper:
            helper.shutdown()
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        kill_leftovers()
        logf.close()
        with open(os.path.join(a.out, a.name + '.json'), 'w') as f:
            json.dump(result, f, indent=1)
        for ph in ('idle', 'nav'):
            if ph in result:
                r = result[ph]
                print(f"[profiler] {a.name} {ph}: nodes {r['total_cores']:.3f} cores "
                      f"(system {r['system_busy_cores']:.2f}/{r['ncpu']}), "
                      f"PSS {r['total_pss_mb']:.0f} MB, msgs {r.get('msgs')}, jetson_hz {r.get('jetson_hz')}", flush=True)
                for k, v in sorted(r['procs'].items(), key=lambda kv: -kv[1]['cores'])[:25]:
                    print(f"    {k:32s} {v['cores']*100:6.1f}%  p95 {v['p95']*100:6.1f}%  "
                          f"PSS {(v['pss_kb'] or 0)/1024:6.1f} MB  thr {v['threads_total']}", flush=True)
                if ph == 'nav':
                    print('    nav log:', r.get('log'), 'nonzero_cmd', r.get('nonzero_cmd'),
                          'last_cmd', r.get('last_cmd'), flush=True)


if __name__ == '__main__':
    main()
