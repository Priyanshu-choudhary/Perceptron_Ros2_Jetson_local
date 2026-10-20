"""Localization error against ground truth: interpolation and statistics, no ROS.

The EKFs publish at their own 30 Hz on the sim clock, which does not line up
with the 20 Hz simulation ticks, so ground truth is interpolated to each
estimate's stamp rather than matched to the nearest tick (at 10 m/s, half a
tick of mismatch is already 25 cm of fake error).
"""

import bisect
import math

import numpy as np


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class GroundTruthBuffer:
    """Time-ordered (t, x, y, yaw) samples with linear interpolation."""

    def __init__(self, max_age=30.0):
        self.t, self.x, self.y, self.yaw = [], [], [], []
        self.max_age = max_age

    def add(self, t, x, y, yaw):
        if self.t and t <= self.t[-1]:
            return
        self.t.append(t)
        self.x.append(x)
        self.y.append(y)
        self.yaw.append(yaw)
        while self.t and self.t[-1] - self.t[0] > self.max_age:
            for lst in (self.t, self.x, self.y, self.yaw):
                lst.pop(0)

    def at(self, t):
        """(x, y, yaw) at time t, or None outside the buffered span."""
        if not self.t or t < self.t[0] or t > self.t[-1]:
            return None
        i = bisect.bisect_left(self.t, t)
        if self.t[i] == t:
            return self.x[i], self.y[i], self.yaw[i]
        t0, t1 = self.t[i - 1], self.t[i]
        a = (t - t0) / (t1 - t0)
        dyaw = wrap_angle(self.yaw[i] - self.yaw[i - 1])
        return (self.x[i - 1] + a * (self.x[i] - self.x[i - 1]),
                self.y[i - 1] + a * (self.y[i] - self.y[i - 1]),
                wrap_angle(self.yaw[i - 1] + a * dyaw))


class ErrorStats:
    """Accumulates position/heading errors and distance travelled for one estimate topic."""

    def __init__(self):
        self.pos, self.head, self.t = [], [], []
        self.distance = 0.0
        self._last_gt = None

    def add(self, t, est, gt):
        ex, ey = est[0] - gt[0], est[1] - gt[1]
        self.pos.append(math.hypot(ex, ey))
        self.head.append(abs(wrap_angle(est[2] - gt[2])))
        self.t.append(t)
        if self._last_gt is not None:
            self.distance += math.hypot(gt[0] - self._last_gt[0], gt[1] - self._last_gt[1])
        self._last_gt = gt

    def summary(self):
        if not self.pos:
            return None
        p, h = np.asarray(self.pos), np.degrees(np.asarray(self.head))
        span = self.t[-1] - self.t[0]
        return {
            'samples': len(p),
            'rate_hz': (len(p) - 1) / span if span > 0 else float('nan'),
            'distance_m': self.distance,
            'pos_rmse_m': float(np.sqrt(np.mean(p ** 2))),
            'pos_max_m': float(p.max()),
            'pos_final_m': float(p[-1]),
            'drift_pct': 100.0 * float(p[-1]) / self.distance if self.distance > 1.0 else float('nan'),
            'head_rmse_deg': float(np.sqrt(np.mean(h ** 2))),
            'head_max_deg': float(h.max()),
        }


def format_summary(name, s):
    if s is None:
        return '%-18s no samples' % name
    return ('%-18s %6d samples %5.1f Hz  dist %7.1f m | pos RMSE %6.2f m  max %6.2f m  '
            'final %6.2f m  drift %5.2f %% | heading RMSE %5.2f deg  max %5.2f deg'
            % (name, s['samples'], s['rate_hz'], s['distance_m'], s['pos_rmse_m'], s['pos_max_m'],
               s['pos_final_m'], s['drift_pct'], s['head_rmse_deg'], s['head_max_deg']))
