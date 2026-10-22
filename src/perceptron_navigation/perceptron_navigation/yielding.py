"""Speed limit that yields to predicted motion (no ROS).

Nav2 replans around a pedestrian, but a person crossing the road keeps walking into the new
path, and the controller's own collision check only looks a few metres along its arc. Stage 7
already predicts where each moving object will be over the next 2 s, so: for every predicted
position that falls inside the robot's corridor (its planned path, widened by half its width,
the object's radius and a margin), the robot must not have its nose within `gap` of that point
at that moment. That caps the speed at

    v <= (s_conflict - nose - gap) / tau          arrive only once it has had time to clear
    v <= sqrt(2 * decel * (s_conflict - nose - gap))   and always be able to stop short of it

and the smallest cap over all conflicts is the limit (None when nothing conflicts). The
robot's own position along the corridor is s = 0 (rear axle / base_footprint).

Predictions end at 2 s, but a person needs ~4 s to cross a car's corridor at 0.8 m/s: judged on
2 s alone, the robot was let in at the 2 s mark and crawled into the crossing line while the
person was still in it (Stage 11 nav_people). So each path is extended at its final velocity
for up to `extend` s, until the object has left the corridor.
"""

import math

import numpy as np


class Corridor:
    """The robot's path ahead as a polyline from the robot, with arc length."""

    def __init__(self, points, max_length=30.0):
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(p) >= 2:
            keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
            p = p[keep]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] if len(p) > 1 else np.zeros(len(p))
        cut = int(np.searchsorted(s, max_length, side='right'))
        self.p, self.s = p[:max(cut, 2)], s[:max(cut, 2)]

    @classmethod
    def straight(cls, x, y, yaw, length=30.0):
        return cls([(x, y), (x + length * math.cos(yaw), y + length * math.sin(yaw))], length)

    def locate(self, xy):
        """(arc length of the nearest corridor point, distance to it)."""
        a, b = self.p[:-1], self.p[1:]
        d = b - a
        L2 = np.maximum((d ** 2).sum(axis=1), 1e-12)
        t = np.clip(((np.asarray(xy) - a) * d).sum(axis=1) / L2, 0.0, 1.0)
        q = a + d * t[:, None]
        dist = np.linalg.norm(q - xy, axis=1)
        k = int(np.argmin(dist))
        return float(self.s[k] + t[k] * math.sqrt(L2[k])), float(dist[k])


def _extended(path, dt, extend):
    """Path continued at its last velocity until `extend` s (no change if it already is that long)."""
    n = int(round(extend / dt)) + 1 if dt > 0 else len(path)
    if len(path) < 2 or len(path) >= n:
        return path
    v = path[-1] - path[-2]
    k = np.arange(1, n - len(path) + 1)[:, None]
    return np.vstack([path, path[-1] + v * k])


def speed_cap(corridor, objects, half_width, nose, gap=1.5, margin=0.5, decel=2.0, extend=6.0):
    """Speed limit (m/s) from predicted object paths, or None.

    objects: [{'path': (N, 2) positions, 'path_dt': s between them (path[0] = now), 'radius': m}]
    Returns (cap, conflict) where conflict = (tau, s, distance off the corridor) of the binding one.
    """
    best, why = None, None
    for ob in objects:
        path = _extended(np.asarray(ob['path'], dtype=float).reshape(-1, 2), ob['path_dt'], extend)
        reach = half_width + ob.get('radius', 0.4) + margin
        for k, xy in enumerate(path):
            s, off = corridor.locate(xy)
            if off > reach or s <= nose or s >= corridor.s[-1] - 1e-6:
                continue      # beside or behind the nose (stopping cannot help), or beyond the corridor
            room = s - nose - gap
            tau = k * ob['path_dt']
            stop = math.sqrt(2.0 * decel * room) if room > 0 else 0.0
            cap = stop if tau <= 0 else min(stop, max(room, 0.0) / tau)
            if best is None or cap < best:
                best, why = cap, (tau, s, off)
    return best, why


class Limiter:
    """Smooths the cap: drops at once, rises at `rise` m/s per second (no stop-go chatter)."""

    def __init__(self, rise=1.0, release=None):
        self.rise = rise
        self.release = release          # cap above this = no limit
        self.v = None

    def update(self, cap, dt):
        if cap is None:
            if self.v is None:
                return None
            cap = math.inf
        if self.v is None or cap < self.v:
            self.v = cap
        else:
            self.v = min(cap, self.v + self.rise * dt)
        if self.release is not None and self.v >= self.release:
            self.v = None
        return self.v
