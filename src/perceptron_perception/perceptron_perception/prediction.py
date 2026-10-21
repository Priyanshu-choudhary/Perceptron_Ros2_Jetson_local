"""Short-horizon trajectory prediction for tracked objects (roadmap Stage 7), no ROS.

Input is one Stage 6 track: [x, y, vx, vy] in odom with its full 4x4 covariance.
Output is the predicted centre and 2D covariance at each horizon plus a dense path,
for the Stage 8 costmap and RViz.

THE PROBLEM: ERROR AMPLIFICATION
A position predicted t seconds ahead carries the velocity error times t: +/-0.3 m/s
is +/-0.6 m at 2 s. A naive constant-velocity extrapolation therefore (a) makes parked
cars creep, (b) throws noisy-but-slow objects far away, and (c) draws a single line
with no notion of how wrong it can be. Each step below addresses one of those.

1. STATIC VS DYNAMIC GATING (with hysteresis)
   A track is predicted as STATIC -- stays where it is, uncertainty = its position
   uncertainty -- until its `moving` flag (speed > max(min_speed, 2 sigma), Stage 6)
   has held for on_frames in a row; it drops back to static only after off_frames
   without it. The asymmetry keeps a parked car from flickering into motion on one
   noisy frame, while a pedestrian who starts walking is picked up within ~0.15 s.
   Class 'other' (cones, boxes) needs a higher speed to count as dynamic at all.

2. VELOCITY SHRINKAGE
   The velocity used is v * |v|^2 / (|v|^2 + tr(Sigma_v)/2): a well-measured velocity
   passes almost unchanged, one no bigger than its own noise is pulled towards zero.
   This is the minimum-mean-square-error scaling for a noisy estimate of an unknown
   speed, and it is what stops a 0.3 +/- 0.3 m/s track from drifting 0.6 m.

3. MOTION MODEL
   CV for pedestrians and anything without a clearly measured turn. CTRV for vehicles
   whose heading history shows a turn rate |w| > max(min_yaw_rate, yaw_sigmas * sigma_w): a car
   rounding a corner is then predicted round the corner rather than off the road.
   The turn rate comes from a least-squares fit of the velocity heading over the last
   yaw_window seconds (YawRateEstimator), because the tracker's own state has none.

4. PHYSICAL LIMITS
   speed  <= class max (pedestrian 3 m/s, vehicle 20 m/s)
   |w|    <= min(v / R_min, a_lat_max / v): a car cannot turn tighter than its minimum
             radius, nor faster than its tyres' lateral grip allows
   The covariance is capped so its 2-sigma ellipse never exceeds what the object could
   physically reach in t (distance |v| t + a_max t^2 / 2 from the prediction).

5. UNCERTAINTY EXPANSION
   Sigma(t) = F(t) Sigma F(t)^T + Q(t), F the constant-velocity transition, Q white-
   noise acceleration with the class's sigma_a (q t^3/3 on position). CTRV adds the
   turn-rate uncertainty across the direction of travel: (v t^2 / 2)^2 sigma_w^2.
   For a static prediction only the position covariance.
   Every prediction also gets the class's pos_floor: the tracked centre carries a
   SYSTEMATIC offset (the lidar sees only an object's near side, Stages 5-6) that the
   tracker's covariance, built from random noise, knows nothing about. Without it the
   ellipses were measured too tight: 71 % of true positions inside the 95 % ellipse
   for static objects, 87-89 % for walkers.
"""

import math
from collections import deque

import numpy as np

STATIC, CV, CTRV = 0, 1, 2
MODEL_NAMES = {STATIC: 'static', CV: 'cv', CTRV: 'ctrv'}
VEHICLES = ('car', 'truck', 'van', 'bus', 'motorcycle', 'bicycle', 'vehicle')

DEFAULT_LIMITS = {
    'pedestrian': {'v_max': 3.0, 'a_max': 3.0, 'sigma_a': 1.5, 'radius': 0.4, 'min_dynamic_speed': 0.3,
                   'pos_floor': 0.15},
    'vehicle': {'v_max': 20.0, 'a_max': 4.0, 'sigma_a': 2.0, 'radius': 2.4, 'min_dynamic_speed': 0.5,
                'r_min': 5.0, 'a_lat_max': 4.0, 'pos_floor': 1.2},     # measured centroid offset
    'other': {'v_max': 3.0, 'a_max': 1.0, 'sigma_a': 0.3, 'radius': 0.4, 'min_dynamic_speed': 1.0,
              'pos_floor': 0.15},
}


def class_group(name):
    if name == 'pedestrian':
        return 'pedestrian'
    return 'vehicle' if name in VEHICLES else 'other'


def shrink_velocity(v, Pv):
    """MMSE scaling of a noisy velocity towards zero (see module docstring)."""
    s2 = float(np.dot(v, v))
    noise = float(np.trace(Pv)) / 2.0
    return v * (s2 / (s2 + noise)) if s2 > 0 else v * 0.0


def cap_yaw_rate(w, speed, lim):
    if speed < 1e-3:
        return 0.0
    w_max = min(speed / lim.get('r_min', 5.0), lim.get('a_lat_max', 4.0) / speed)
    return float(np.clip(w, -w_max, w_max))


def cap_covariance(S, reach):
    """Shrink a 2x2 covariance so its 2-sigma ellipse fits in a disc of radius `reach`."""
    vals, vecs = np.linalg.eigh(S)
    limit = (reach / 2.0) ** 2
    vals = np.minimum(vals, max(limit, 1e-6))
    return vecs @ np.diag(vals) @ vecs.T


class YawRateEstimator:
    """Turn rate from the heading of a track's velocity over the last `window` seconds."""

    def __init__(self, window=1.0, min_speed=1.5):
        self.window, self.min_speed = window, min_speed
        self.hist = deque()

    def add(self, t, vx, vy):
        if math.hypot(vx, vy) < self.min_speed:
            self.hist.clear()                    # heading of a near-stopped object is noise
            return
        h = math.atan2(vy, vx)
        if self.hist:                            # unwrap against the previous sample
            prev = self.hist[-1][1]
            h = prev + (h - prev + math.pi) % (2 * math.pi) - math.pi
        self.hist.append((t, h))
        while self.hist and t - self.hist[0][0] > self.window:
            self.hist.popleft()

    def estimate(self):
        """(yaw_rate, sigma) or (0, inf) with too little history."""
        if len(self.hist) < 5:
            return 0.0, float('inf')
        t = np.array([s[0] for s in self.hist])
        h = np.array([s[1] for s in self.hist])
        t = t - t.mean()
        denom = float(np.dot(t, t))
        if denom < 1e-6:
            return 0.0, float('inf')
        w = float(np.dot(t, h - h.mean()) / denom)
        resid = h - h.mean() - w * t
        sigma = math.sqrt(max(float(np.dot(resid, resid)) / max(len(t) - 2, 1), 1e-8) / denom)
        return w, sigma


class MotionGate:
    """Static/dynamic hysteresis for one track."""

    def __init__(self, on_frames=3, off_frames=10):
        self.on, self.off = on_frames, off_frames
        self.dynamic, self.count = False, 0

    def update(self, moving):
        if moving != self.dynamic:
            self.count += 1
            if self.count >= (self.on if moving else self.off):
                self.dynamic, self.count = moving, 0
        else:
            self.count = 0
        return self.dynamic


def predict(state, P, cls, dynamic, horizons, path_dt=0.1, yaw=(0.0, float('inf')), limits=None,
            min_yaw_rate=0.1, yaw_sigmas=3.0):
    """Predict one track. Returns dict(model, speed, yaw_rate, radius, means (H,2),
    covs (H,2,2), path (N,2))."""
    lim = (limits or DEFAULT_LIMITS)[class_group(cls)]
    pos = np.asarray(state[:2], float)
    P = np.asarray(P, float).reshape(4, 4)
    horizons = [float(h) for h in horizons]
    t_end = max(horizons) if horizons else 0.0
    path_t = np.arange(0.0, t_end + 1e-9, path_dt)

    v = np.asarray(state[2:4], float)
    speed_raw = float(np.linalg.norm(v))
    if not dynamic or speed_raw < lim['min_dynamic_speed']:
        S = P[:2, :2] + np.eye(2) * lim['pos_floor'] ** 2
        return {'model': STATIC, 'speed': 0.0, 'yaw_rate': 0.0, 'radius': lim['radius'],
                'means': np.tile(pos, (len(horizons), 1)), 'covs': np.array([S] * len(horizons)),
                'path': np.tile(pos, (len(path_t), 1))}

    v = shrink_velocity(v, P[2:, 2:])
    speed = float(np.linalg.norm(v))
    if speed > lim['v_max']:
        v = v / speed * lim['v_max']
        speed = lim['v_max']
    heading = math.atan2(v[1], v[0])

    model, w = CV, 0.0
    w_raw, w_sigma = yaw
    if class_group(cls) == 'vehicle' and abs(w_raw) > max(min_yaw_rate, yaw_sigmas * w_sigma):
        w = cap_yaw_rate(w_raw, speed, lim)
        model = CTRV

    def mean_at(t):
        if model == CTRV and abs(w) > 1e-6:
            return pos + speed / w * np.array([math.sin(heading + w * t) - math.sin(heading),
                                                math.cos(heading) - math.cos(heading + w * t)])
        return pos + v * t

    q = lim['sigma_a'] ** 2
    means, covs = [], []
    for t in horizons:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = t
        Q = np.zeros((4, 4))
        Q[:2, :2] = np.eye(2) * q * t ** 3 / 3.0
        Q[:2, 2:] = Q[2:, :2] = np.eye(2) * q * t ** 2 / 2.0
        Q[2:, 2:] = np.eye(2) * q * t
        S = (F @ P @ F.T + Q)[:2, :2]
        if model == CTRV and math.isfinite(w_sigma):
            lat = np.array([-math.sin(heading + w * t), math.cos(heading + w * t)])
            S = S + np.outer(lat, lat) * (speed * t * t / 2.0) ** 2 * w_sigma ** 2
        reach = (speed * t + lim['a_max'] * t * t / 2.0 +
                 2.0 * math.sqrt(max(P[0, 0], P[1, 1]) + lim['pos_floor'] ** 2))
        means.append(mean_at(t))
        covs.append(cap_covariance(S + np.eye(2) * lim['pos_floor'] ** 2, reach))
    return {'model': model, 'speed': speed, 'yaw_rate': w, 'radius': lim['radius'],
            'means': np.array(means), 'covs': np.array(covs),
            'path': np.array([mean_at(t) for t in path_t])}
