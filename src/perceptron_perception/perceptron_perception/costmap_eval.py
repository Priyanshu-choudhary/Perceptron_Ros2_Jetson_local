"""Scoring the Stage 8 costmap against CARLA ground truth (no ROS).

  obstacle recall   every actor the lidar hit (>= 3 semantic points) within range: is
                    its true footprint covered by a lethal / inscribed cell (>= 99 on
                    the 0-100 scale) of the combined map?
  false obstacles   lethal cells of the obstacle layer whose lidar returns this frame
                    were ALL ground (road, sidewalk, terrain...): the costmap blocking
                    free space
  dynamic coverage  for each moving actor at t, the risk grid of time t looked up at
                    where the actor really was at t + 1 s and t + 2 s
  flicker           share of known cells near the robot whose lethal state toggles
                    between consecutive updates (the grids are aligned for the shift)
"""

import math

import numpy as np


def grid_array(msg):
    """OccupancyGrid -> (values [nx, ny] int8, origin xy, resolution)."""
    a = np.frombuffer(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width).T
    return a, np.array([msg.info.origin.position.x, msg.info.origin.position.y]), msg.info.resolution


def cells_in_box(origin, res, shape, centre, yaw, size_xy):
    """(i, j) index arrays of cells whose centres lie inside an oriented rectangle."""
    hx, hy = size_xy[0] / 2.0, size_xy[1] / 2.0
    r = math.hypot(hx, hy)
    i0 = max(int((centre[0] - r - origin[0]) / res), 0)
    j0 = max(int((centre[1] - r - origin[1]) / res), 0)
    i1 = min(int((centre[0] + r - origin[0]) / res) + 1, shape[0])
    j1 = min(int((centre[1] + r - origin[1]) / res) + 1, shape[1])
    if i0 >= i1 or j0 >= j1:
        return np.zeros(0, int), np.zeros(0, int)
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
    x = origin[0] + (ii + 0.5) * res - centre[0]
    y = origin[1] + (jj + 0.5) * res - centre[1]
    c, s = math.cos(yaw), math.sin(yaw)
    inside = (np.abs(c * x + s * y) <= hx + res / 2) & (np.abs(-s * x + c * y) <= hy + res / 2)
    return ii[inside], jj[inside]


def value_at(grid, origin, res, xy):
    i = int((xy[0] - origin[0]) / res)
    j = int((xy[1] - origin[1]) / res)
    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
        return int(grid[i, j])
    return None


def flicker(prev, cur):
    """Share of cells known in both maps whose lethal state differs; maps = (grid, origin, res)."""
    (g0, o0, res), (g1, o1, _) = prev, cur
    shift = np.rint((o1 - o0) / res).astype(int)
    nx, ny = g1.shape
    a = g0[max(shift[0], 0):nx + min(shift[0], 0), max(shift[1], 0):ny + min(shift[1], 0)]
    b = g1[max(-shift[0], 0):nx + min(-shift[0], 0), max(-shift[1], 0):ny + min(-shift[1], 0)]
    known = (a >= 0) & (b >= 0)
    if not known.any():
        return float('nan')
    return float(((a >= 99) != (b >= 99))[known].mean())


class CostmapScores:

    def __init__(self):
        self.recall = {}            # (group, moving) -> [actors, covered]
        self.tp = self.fp = 0
        self.fp_tags = {}           # semantic tag -> ground points seen in false-obstacle cells
        self.dyn = {1.0: [], 2.0: []}
        self.flicker = []
        self.stats = []
        self.frames = 0

    def add_recall(self, group, moving, covered):
        r = self.recall.setdefault((group, moving), [0, 0])
        r[0] += 1
        r[1] += int(covered)

    def summary(self):
        s = {'frames': self.frames,
             'recall': {k: (n, 100.0 * c / n) for k, (n, c) in sorted(self.recall.items()) if n},
             'false_obstacle_pct': 100.0 * self.fp / max(self.tp + self.fp, 1),
             'lethal_cells_scored': self.tp + self.fp,
             'fp_tags': {k: 100.0 * v / max(sum(self.fp_tags.values()), 1)
                         for k, v in sorted(self.fp_tags.items(), key=lambda kv: -kv[1])[:4]},
             'dynamic': {h: (len(v), 100.0 * np.mean([x > 5 for x in v]) if v else float('nan'),
                             float(np.mean(v)) if v else float('nan')) for h, v in self.dyn.items()},
             'flicker_pct': 100.0 * float(np.nanmean(self.flicker)) if self.flicker else float('nan')}
        if self.stats:
            a = np.array(self.stats)
            s['ms_mean'], s['ms_p95'] = float(a[:, 0].mean()), float(np.percentile(a[:, 0], 95))
            s['age_mean'] = float(a[:, 5].mean())
        return s


def format_summary(s, title):
    lines = [title + ' (%d frames)' % s['frames']]
    for (g, moving), (n, pct) in s['recall'].items():
        lines.append('  obstacle recall  %-10s %-7s %6d actor-frames  %5.1f %% covered by lethal/inscribed'
                     % (g, 'moving' if moving else 'static', n, pct))
    lines.append('  false obstacles  %.2f %% of %d scored lethal cells saw only ground this frame'
                 % (s['false_obstacle_pct'], s['lethal_cells_scored']))
    if s.get('fp_tags'):
        from perceptron_perception.cloud_eval import TAG_NAMES
        lines.append('                   ground classes in those cells: ' + ', '.join(
            '%s %.0f %%' % (TAG_NAMES.get(k, k), v) for k, v in s['fp_tags'].items()))
    for h, (n, pct, mean) in s['dynamic'].items():
        lines.append('  dynamic risk     at true position %.0f s later: %5.1f %% inside the risk layer (n=%d, mean cost %.0f/100)'
                     % (h, pct, n, mean))
    lines.append('  flicker          %.2f %% of known cells toggle lethal per update' % s['flicker_pct'])
    if 'ms_mean' in s:
        lines.append('  latency          update %.1f ms mean, p95 %.1f ms; data age at publish %.0f ms'
                     % (s['ms_mean'], s['ms_p95'], s['age_mean']))
    return '\n'.join(lines)
