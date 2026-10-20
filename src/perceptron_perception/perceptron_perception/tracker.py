"""Multi-object tracking in a fixed frame (roadmap Stage 6), no ROS.

Detections arrive already moved into odom (the node does that through TF at
each detection's stamp), so the robot's own motion is gone before anything
here runs: a parked car keeps the same odom position however the robot drives,
and its estimated velocity converges to zero. That is the whole of ego-motion
compensation, and why the Stage 3 EKF is a dependency.

PER TRACK: a constant-velocity Kalman filter on [x, y, vx, vy]
  predict  F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],
           Q = white-noise acceleration with the class's sigma_a (people turn
           and stop quickly, cars do not)
  update   z = [x, y], R = the detection's own covariance + a class floor
  height   exponentially smoothed separately; a ground robot plans in 2D

ASSOCIATION: Mahalanobis distance of each detection to each predicted track
(chi-square gate, 2 dof), only between compatible classes, solved with the
Hungarian algorithm (scipy linear_sum_assignment) so two people walking side
by side cannot both claim the same track.

MANOEUVRES AND DUPLICATES
  A person who stops or turns round leaves the constant-velocity prediction
  behind, the detection falls outside the gate, and a naive tracker starts a
  second track next to the first; detections then flicker between the two and
  the id changes. Three guards:
  * tentative tracks pay tentative_penalty in the assignment, so a detection in
    both gates goes to the established track
  * a second pass re-acquires: a detection within the class's reacquire radius
    of a missed track's LAST MEASURED position re-attaches to it, and the
    track's velocity variance is reset so it re-learns the new motion
  * after updating, two tracks of one class closer than the prune distance are
    one object: the younger is dropped

LIFECYCLE
  tentative  born from an unmatched detection; confirmed after confirm_hits
             hits, deleted after tentative_misses misses (sensor flicker
             never becomes a "ghost" obstacle)
  confirmed  updated this frame
  coasting   confirmed but missed: predicted forward, kept up to
             max_coast_time seconds (a person behind a pole for a few frames
             keeps their id), then deleted

VELOCITY: a new track starts with zero velocity and a large velocity
variance, so the first few updates set it. `moving` is speed > max(min_speed,
2 sigma of the speed estimate), so a noisy-but-parked car is not "moving".

NEAR-FACE BIAS: the lidar only sees an object's near side (Stage 5), so each
detection is pushed back along the viewing ray by half the class's depth
before it reaches the filter (class_depth_offset).
"""

import itertools
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

VEHICLES = ('car', 'truck', 'van', 'bus', 'motorcycle', 'bicycle', 'vehicle')

# per class group: acceleration noise (m/s^2), measurement floor (m), depth offset (m),
# re-acquire radius (m), duplicate prune distance (m)
DEFAULT_CLASS_PARAMS = {
    'pedestrian': {'sigma_a': 3.0, 'r_floor': 0.15, 'depth_offset': 0.15, 'reacquire': 1.0, 'prune': 0.4},
    'vehicle': {'sigma_a': 3.0, 'r_floor': 0.35, 'depth_offset': 1.0, 'reacquire': 2.5, 'prune': 1.2},
    'other': {'sigma_a': 0.5, 'r_floor': 0.15, 'depth_offset': 0.1, 'reacquire': 0.6, 'prune': 0.3},
}

TENTATIVE, CONFIRMED, COASTING = 0, 1, 2


def class_group(name):
    if name == 'pedestrian':
        return 'pedestrian'
    return 'vehicle' if name in VEHICLES else 'other'


def correct_near_face(xy, sensor_xy, offset):
    """Push a detection `offset` metres further along the ray from the sensor."""
    d = np.asarray(xy, float) - np.asarray(sensor_xy, float)
    n = np.linalg.norm(d)
    return np.asarray(xy, float) if n < 1e-6 else np.asarray(xy, float) + d / n * offset


class Track:
    _ids = itertools.count(1)

    def __init__(self, det, t, params, vel_var):
        self.id = next(Track._ids)
        self.x = np.array([det['xy'][0], det['xy'][1], 0.0, 0.0])
        R = det['R']
        self.P = np.diag([R[0, 0], R[1, 1], vel_var, vel_var])
        self.z = det['z']
        self.size = det['size']
        self.votes = {det['class']: 1}
        self.group = class_group(det['class'])
        self.params = params
        self.t_first = self.t_last_update = self.t = t
        self.hits, self.misses, self.state = 1, 0, TENTATIVE
        self.source = det.get('id', '')
        self.last_xy = np.asarray(det['xy'], float)
        self.vel_var = vel_var

    @property
    def class_id(self):
        return max(self.votes, key=self.votes.get)

    @property
    def class_confidence(self):
        return self.votes[self.class_id] / sum(self.votes.values())

    def predict(self, t):
        dt = max(0.0, t - self.t)
        if dt > 0:
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            q = self.params['sigma_a'] ** 2
            G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + G @ G.T * q
        self.t = t

    def innovation(self, det):
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        y = np.asarray(det['xy']) - H @ self.x
        S = H @ self.P @ H.T + det['R']
        return y, S, H

    def mahalanobis2(self, det):
        y, S, _ = self.innovation(det)
        return float(y @ np.linalg.solve(S, y))

    def update(self, det, t, manoeuvre=False):
        if manoeuvre:
            # re-acquired after leaving the gate: forget the old velocity, keep the id
            self.P[2:, 2:] = np.eye(2) * self.vel_var
            self.P[2:, :2] = self.P[:2, 2:] = 0.0
        y, S, H = self.innovation(det)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.P = (self.P + self.P.T) / 2
        self.z = 0.7 * self.z + 0.3 * det['z']
        self.size = det['size']
        self.votes[det['class']] = self.votes.get(det['class'], 0) + 1
        self.hits += 1
        self.misses = 0
        self.t_last_update = t
        self.source = det.get('id', '')
        self.last_xy = np.asarray(det['xy'], float)

    @property
    def speed(self):
        return float(math.hypot(self.x[2], self.x[3]))

    def speed_sigma(self):
        v = self.x[2:]
        n = np.linalg.norm(v)
        if n < 1e-6:
            return float(math.sqrt(max(self.P[2, 2], self.P[3, 3])))
        u = v / n
        return float(math.sqrt(u @ self.P[2:, 2:] @ u))

    def moving(self, min_speed):
        return self.speed > max(min_speed, 2.0 * self.speed_sigma())


class Tracker:

    def __init__(self, class_params=None, gate_chi2=9.21, confirm_hits=3, tentative_misses=2,
                 max_coast_time=1.0, init_vel_var=None, min_speed=0.3, tentative_penalty=2.0):
        self.class_params = class_params or DEFAULT_CLASS_PARAMS
        self.gate = gate_chi2
        self.confirm_hits = confirm_hits
        self.tentative_misses = tentative_misses
        self.max_coast = max_coast_time
        self.init_vel_var = init_vel_var or {'pedestrian': 2.0 ** 2, 'vehicle': 10.0 ** 2, 'other': 1.0}
        self.min_speed = min_speed
        self.tentative_penalty = tentative_penalty
        self.tracks = []

    def make_detection(self, xy, z, cls, cov_xy, size, det_id='', sensor_xy=None):
        """Normalise one detection: near-face correction and measurement noise."""
        p = self.class_params[class_group(cls)]
        if sensor_xy is not None:
            xy = correct_near_face(xy, sensor_xy, p['depth_offset'])
        R = np.asarray(cov_xy, float).reshape(2, 2) + np.eye(2) * p['r_floor'] ** 2
        return {'xy': np.asarray(xy, float), 'z': float(z), 'class': cls, 'R': R,
                'size': np.asarray(size, float), 'id': det_id}

    def step(self, detections, t):
        """Predict all tracks to t, associate, update, manage lifecycle. Returns the tracks."""
        for tr in self.tracks:
            tr.predict(t)
            tr.source = ''

        matched_t, matched_d = set(), set()
        if self.tracks and detections:
            big = 1e6
            cost = np.full((len(self.tracks), len(detections)), big)
            for i, tr in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    if class_group(d['class']) != tr.group:
                        continue
                    m2 = tr.mahalanobis2(d)
                    if m2 <= self.gate:
                        cost[i, j] = m2 + (self.tentative_penalty if tr.state == TENTATIVE else 0.0)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < big:
                    self.tracks[i].update(detections[j], t)
                    matched_t.add(i)
                    matched_d.add(j)

        # second pass: re-acquire missed established tracks from their last measured position
        free_t = [i for i, tr in enumerate(self.tracks) if i not in matched_t and tr.state != TENTATIVE]
        free_d = [j for j in range(len(detections)) if j not in matched_d]
        if free_t and free_d:
            big = 1e6
            cost = np.full((len(free_t), len(free_d)), big)
            for a, i in enumerate(free_t):
                tr = self.tracks[i]
                for b, j in enumerate(free_d):
                    d = detections[j]
                    if class_group(d['class']) != tr.group:
                        continue
                    dist = float(np.linalg.norm(d['xy'] - tr.last_xy))
                    if dist <= tr.params['reacquire']:
                        cost[a, b] = dist
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] < big:
                    i, j = free_t[a], free_d[b]
                    self.tracks[i].update(detections[j], t, manoeuvre=True)
                    matched_t.add(i)
                    matched_d.add(j)

        survivors = []
        for i, tr in enumerate(self.tracks):
            if i in matched_t:
                if tr.state == TENTATIVE and tr.hits >= self.confirm_hits:
                    tr.state = CONFIRMED
                elif tr.state == COASTING:
                    tr.state = CONFIRMED
                survivors.append(tr)
                continue
            tr.misses += 1
            if tr.state == TENTATIVE:
                if tr.misses < self.tentative_misses:
                    survivors.append(tr)
                continue
            tr.state = COASTING
            if t - tr.t_last_update <= self.max_coast:
                survivors.append(tr)
        for j, d in enumerate(detections):
            if j not in matched_d:
                g = class_group(d['class'])
                survivors.append(Track(d, t, self.class_params[g], self.init_vel_var[g]))
        self.tracks = self._prune(survivors)
        return self.tracks

    @staticmethod
    def _prune(tracks):
        """Two tracks of one class on the same spot are one object: keep the older."""
        order = sorted(tracks, key=lambda tr: (-tr.hits, tr.id))
        kept = []
        for tr in order:
            if any(k.group == tr.group and
                   np.linalg.norm(k.x[:2] - tr.x[:2]) < min(k.params['prune'], tr.params['prune'])
                   for k in kept):
                continue
            kept.append(tr)
        return sorted(kept, key=lambda tr: tr.id)
