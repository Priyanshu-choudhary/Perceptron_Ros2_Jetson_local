#!/usr/bin/env python3
"""Check the simulated rover sensors against ground truth and the TF tree (Stages 1-2).

    ros2 run perceptron_carla sensor_check --duration 30

Listens for --duration seconds of wall time, then prints:
  * rate of every robot-equivalent topic
  * wheel odometry: speed scale and residual, yaw-rate residual vs truth
  * IMU: heading offset/noise vs truth, gyro z residual, gravity on z
  * GNSS: fix displacement vs true displacement (relative to the first fix)
  * TF: every child frame and its parent; FAILS if any frame has two parents or
    if the sensor frames do not chain up to carla_world
Exit code 1 on any failure, so it can gate a scenario run.
"""

import argparse
import math
import sys
import time
from collections import defaultdict

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix, PointCloud2, Image
from tf2_msgs.msg import TFMessage

from perceptron_carla import loc_metrics as lm
from perceptron_carla import sim_sensors as ss

REQUIRED_CHAIN = ['camera_optical_link', 'camera_link', 'lidar3d_link', 'imu_link', 'gps_link',
                  'base_link', 'base_footprint', 'odom', 'map']


class SensorCheck(Node):
    def __init__(self):
        super().__init__('sensor_check')
        self.t0 = time.time()
        self.counts = defaultdict(list)
        self.gt = lm.GroundTruthBuffer(max_age=600.0)
        self.gt_twist = []
        self.odom, self.imu, self.gps = [], [], []
        self.parents = defaultdict(set)
        latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)

        def count(topic):
            return lambda m: self.counts[topic].append(time.time())

        for topic, typ, qos in (('/carla/camera/image_raw', Image, qos_profile_sensor_data),
                                ('/carla/lidar/points', PointCloud2, qos_profile_sensor_data)):
            self.create_subscription(typ, topic, count(topic), qos)
        self.create_subscription(Odometry, '/carla/ground_truth/ego_odom', self._gt, 50)
        self.create_subscription(Odometry, '/odom', self._odom, 50)
        self.create_subscription(Imu, '/imu/data', self._imu, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, '/gps/fix', self._gps, 50)
        self.create_subscription(TFMessage, '/tf', self._tf, 100)
        self.create_subscription(TFMessage, '/tf_static', self._tf, latched)

    @staticmethod
    def _t(h):
        return h.stamp.sec + h.stamp.nanosec * 1e-9

    def _gt(self, m):
        q = m.pose.pose.orientation
        t = self._t(m.header)
        self.gt.add(t, m.pose.pose.position.x, m.pose.pose.position.y, lm.yaw_from_quat(q.x, q.y, q.z, q.w))
        self.gt_twist.append((t, m.twist.twist.linear.x, m.twist.twist.angular.z))
        self.counts['/carla/ground_truth/ego_odom'].append(time.time())

    def _odom(self, m):
        self.odom.append((self._t(m.header), m.twist.twist.linear.x, m.twist.twist.angular.z,
                          m.header.frame_id, m.child_frame_id))
        self.counts['/odom'].append(time.time())

    def _imu(self, m):
        q = m.orientation
        self.imu.append((self._t(m.header), lm.yaw_from_quat(q.x, q.y, q.z, q.w), m.angular_velocity.z,
                         m.linear_acceleration.z, m.header.frame_id))
        self.counts['/imu/data'].append(time.time())

    def _gps(self, m):
        self.gps.append((self._t(m.header), m.latitude, m.longitude, m.header.frame_id,
                         m.position_covariance[0]))
        self.counts['/gps/fix'].append(time.time())

    def _tf(self, msg):
        for t in msg.transforms:
            self.parents[t.child_frame_id].add(t.header.frame_id)

    # ------------------------------------------------------------------ report
    def report(self):
        ok = True
        out = ['', 'RATES (wall clock)']
        for topic in ('/carla/camera/image_raw', '/carla/lidar/points', '/odom', '/imu/data', '/gps/fix',
                      '/carla/ground_truth/ego_odom'):
            ts = self.counts.get(topic, [])
            rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 else 0.0
            out.append('  %-32s %6.1f Hz  (%d msgs)' % (topic, rate, len(ts)))
            if not ts:
                ok = False
                out.append('    FAIL: nothing received')

        gt_tw = {round(t, 3): (v, w) for t, v, w in self.gt_twist}
        out.append('WHEEL ODOMETRY vs truth')
        pairs = [(v, w, gt_tw[round(t, 3)]) for t, v, w, _, _ in self.odom if round(t, 3) in gt_tw]
        if pairs:
            v = np.array([p[0] for p in pairs])
            w = np.array([p[1] for p in pairs])
            gv = np.array([p[2][0] for p in pairs])
            gw = np.array([p[2][1] for p in pairs])
            moving = np.abs(gv) > 0.5
            scale = float(np.sum(v[moving] * gv[moving]) / np.sum(gv[moving] ** 2)) if moving.any() else float('nan')
            out.append('  frames %s -> %s' % (self.odom[0][3], self.odom[0][4]))
            out.append('  speed scale %.3f (truth 1.000), residual std %.3f m/s, max true speed %.1f m/s'
                       % (scale, float(np.std(v - scale * gv)), float(np.abs(gv).max())))
            out.append('  yaw-rate residual mean %+.4f std %.4f rad/s' % (float(np.mean(w - gw)), float(np.std(w - gw))))
        else:
            ok = False
            out.append('  FAIL: no /odom samples matched to ground truth')

        out.append('IMU vs truth')
        rows = [(lm.wrap_angle(yaw - g[2]), wz, az) for t, yaw, wz, az, _ in self.imu
                for g in [self.gt.at(t)] if g is not None]
        if rows:
            dyaw = np.degrees([r[0] for r in rows])
            out.append('  frame %s' % self.imu[0][4])
            out.append('  heading offset mean %+.2f deg, std %.2f deg (sim adds a constant offset + walk)'
                       % (float(np.mean(dyaw)), float(np.std(dyaw))))
            gyro = [(wz, gt_tw[round(t, 3)][1]) for t, _, wz, _, _ in self.imu if round(t, 3) in gt_tw]
            if gyro:
                g = np.array(gyro)
                out.append('  gyro z residual mean %+.4f std %.4f rad/s; corr with truth %.3f'
                           % (float(np.mean(g[:, 0] - g[:, 1])), float(np.std(g[:, 0] - g[:, 1])),
                              float(np.corrcoef(g[:, 0], g[:, 1])[0, 1]) if np.std(g[:, 1]) > 1e-6 else float('nan')))
            az = np.array([r[2] for r in rows])
            out.append('  accel z mean %.2f m/s^2 (gravity included, like the Pixhack)' % float(az.mean()))
            if abs(float(np.mean(dyaw))) > 10.0:
                ok = False
                out.append('  FAIL: heading is off by more than 10 deg -- convention problem')
        else:
            ok = False
            out.append('  FAIL: no /imu/data samples matched to ground truth')

        out.append('GNSS vs truth (displacement from the first fix)')
        fixes = [(t, la, lo) for t, la, lo, _, _ in self.gps if self.gt.at(t) is not None]
        if len(fixes) > 3:
            t0, la0, lo0 = fixes[0]
            g0 = self.gt.at(t0)
            errs = []
            for t, la, lo in fixes[1:]:
                n = (la - la0) * ss.EARTH_M_PER_DEG
                e = (lo - lo0) * ss.EARTH_M_PER_DEG * math.cos(math.radians(la0))
                g = self.gt.at(t)
                errs.append(math.hypot(e - (g[0] - g0[0]), n - (g[1] - g0[1])))
            errs = np.array(errs)
            out.append('  frame %s, reported std %.2f m' % (self.gps[0][3], math.sqrt(self.gps[0][4])))
            out.append('  displacement error median %.2f m, p95 %.2f m (white noise only; bias cancels)'
                       % (float(np.median(errs)), float(np.percentile(errs, 95))))
            if np.median(errs) > 5.0:
                ok = False
                out.append('  FAIL: GNSS displacement disagrees with truth -- axis/sign problem')
        else:
            ok = False
            out.append('  FAIL: fewer than 4 /gps/fix samples')

        out.append('TF TREE (child <- parent)')
        for child in sorted(self.parents):
            ps = sorted(self.parents[child])
            flag = ''
            if len(ps) > 1:
                ok = False
                flag = '   FAIL: more than one parent'
            out.append('  %-22s <- %s%s' % (child, ', '.join(ps), flag))
        for frame in REQUIRED_CHAIN:
            f, hops = frame, 0
            while f != 'carla_world' and f in self.parents and hops < 20:
                f, hops = next(iter(self.parents[f])), hops + 1
            if f != 'carla_world':
                ok = False
                out.append('  FAIL: %s does not chain up to carla_world (stops at %s)' % (frame, f))
        out.append('RESULT: %s' % ('PASS' if ok else 'FAIL'))
        print('\n'.join(out), flush=True)
        return ok


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cut = argv.index('--ros-args') if '--ros-args' in argv else len(argv)
    p = argparse.ArgumentParser(description='Check simulated rover sensors and TF against ground truth')
    p.add_argument('--duration', type=float, default=30.0, help='seconds of wall time to listen')
    args = p.parse_args(argv[:cut])
    rclpy.init(args=['sensor_check'] + argv[cut:])
    node = SensorCheck()
    end = time.time() + args.duration
    try:
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    ok = node.report()
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
