"""Scoring camera-lidar objects against CARLA ground truth (no ROS).

For every input 2D box (with the CARLA reference detector its id is the actor
id, so the true object is known):

  coverage     did the fusion produce an object with a range for it?
  not hit      ...and if not, did the lidar hit the actor at all (semantic lidar)?
               Not hit is a sensor limit, not an association failure
  correct      >= 50 % of the object's points really belong to that actor, from
               the semantic lidar labels
  range error  estimated nearest-surface range minus the TRUE distance from the
               camera to the closest point of the actor's 3D box
  position     estimated centroid vs the true box centre, in the ground plane
               (biased by up to half the object's depth, see object_fusion.py)
"""

import numpy as np

from perceptron_navigation import camera_projection as cp

RANGE_BINS = (0.0, 10.0, 20.0, 40.0)
VEHICLES = ('car', 'truck', 'van', 'bus', 'motorcycle', 'bicycle', 'vehicle')


def class_group(name):
    if name == 'pedestrian':
        return 'pedestrian'
    return 'vehicle' if name in VEHICLES else 'other'


def closest_point_distance(origin, centre, quat_xyzw, size):
    """Distance from `origin` to the nearest point of an oriented box (0 when inside)."""
    R = cp.quaternion_to_rotation(*quat_xyzw)
    local = R.T @ (np.asarray(origin, float) - np.asarray(centre, float))
    half = np.asarray(size, float) / 2.0
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


class FusionScores:

    def __init__(self):
        self.rows = {}            # (group, bin) -> dict of lists/counters
        self.frames = 0

    def _row(self, group, rng):
        b = int(np.searchsorted(RANGE_BINS, rng, side='right') - 1)
        if b < 0 or b >= len(RANGE_BINS) - 1:
            return None
        return self.rows.setdefault((group, b), {'boxes': 0, 'estimated': 0, 'no_points': 0,
                                                 'correct': 0, 'range_err': [], 'pos_err': [], 'pos_vec': []})

    def add(self, boxes, objects, truth):
        """boxes: [(id, class, has_points_in_box)]; objects: id -> dict(range, xy, purity);
        truth: id -> dict(range (closest point), xy (box centre))."""
        self.frames += 1
        for bid, cls, has_points in boxes:
            gt = truth.get(bid)
            if gt is None:
                continue
            row = self._row(class_group(cls), gt['range'])
            if row is None:
                continue
            row['boxes'] += 1
            obj = objects.get(bid)
            if obj is None:
                row['no_points'] += int(not has_points)
                continue
            row['estimated'] += 1
            if obj.get('purity') is not None and obj['purity'] >= 0.5:
                row['correct'] += 1
            row['range_err'].append(obj['range'] - gt['range'])
            dxy = np.asarray(obj['xy'], float) - np.asarray(gt['xy'], float)
            row['pos_err'].append(float(np.hypot(*dxy)))
            row['pos_vec'].append(dxy)

    def summary(self):
        out = {}
        for (g, b), r in sorted(self.rows.items()):
            e = np.asarray(r['range_err'])
            p = np.asarray(r['pos_err'])
            n, est = r['boxes'], r['estimated']
            out['%s %g-%g m' % (g, RANGE_BINS[b], RANGE_BINS[b + 1])] = {
                'boxes': n,
                'coverage_pct': 100.0 * est / n if n else float('nan'),
                'no_points_pct': 100.0 * r['no_points'] / n if n else float('nan'),
                'correct_pct': 100.0 * r['correct'] / est if est else float('nan'),
                'range_bias': float(e.mean()) if len(e) else float('nan'),
                'range_mae': float(np.abs(e).mean()) if len(e) else float('nan'),
                'range_p95': float(np.percentile(np.abs(e), 95)) if len(e) else float('nan'),
                'pos_mae': float(p.mean()) if len(p) else float('nan'),
                'pos_bias': (np.mean(r['pos_vec'], axis=0) if r['pos_vec'] else np.full(2, np.nan)).tolist(),
            }
        return out


def format_summary(s, title):
    lines = [title,
             '  class / true range        boxes  with range  not hit  correct pts | range err: '
             'bias    MAE    p95 | centroid - box centre: mean |d|  (mean dx, dy)']
    for k, r in s.items():
        lines.append('  %-24s %6d     %5.1f %%  %5.1f %%    %5.1f %%  |          %+5.2f  %5.2f  %5.2f m'
                     ' |  %5.2f m  (%+.2f, %+.2f)' % (k, r['boxes'], r['coverage_pct'], r['no_points_pct'],
                                                   r['correct_pct'], r['range_bias'], r['range_mae'],
                                                   r['range_p95'], r['pos_mae'], *r['pos_bias']))
    return '\n'.join(lines)
