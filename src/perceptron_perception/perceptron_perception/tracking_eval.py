"""Scoring tracks against CARLA ground truth (roadmap Stage 6), no ROS.

Each frame, confirmed and coasting tracks are matched to the true actors BY
POSITION (Hungarian on ground-plane distance within the class's radius), as
CLEAR-MOT does -- not through source_detection_id. A detector that lends one
person's box label to another person's points for a frame (Stage 5 does, when
two people overlap in the image) would otherwise look like a tracker swap
although the track never left its person.

  ID switches  an actor that was on track A is now on track B (counted once per change)
  swapped      a track that has carried more than one actor over its life
  position     |track xy - true box centre xy|
  velocity     |track v - true v| (vector), and speed error, split into static
               (true speed < 0.2 m/s) and moving actors; heading error for movers
               faster than 0.5 m/s
"""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from perceptron_perception.tracker import class_group

MATCH_RADIUS = {'pedestrian': 1.0, 'vehicle': 2.5, 'other': 1.0}


def match_by_position(tracks, truth):
    """[(track, actor id)] pairs: Hungarian on xy distance, same class group, within radius."""
    pairs = []
    for g, radius in MATCH_RADIUS.items():
        tr = [t for t in tracks if class_group(t['class']) == g]
        gt = [(a, v) for a, v in truth.items() if class_group(v['class']) == g]
        if not tr or not gt:
            continue
        d = np.array([[np.hypot(*(t['xy'] - v['xy'])) for _, v in gt] for t in tr])
        cost = np.where(d <= radius, d, 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1e6:
                pairs.append((tr[i], gt[j][0]))
    return pairs


class TrackingScores:

    def __init__(self):
        self.frames = 0
        self.last_track = {}            # actor -> track id last frame it was matched
        self.track_actors = {}          # track id -> set of actors
        self.rows = {}                  # group -> metrics

    def _row(self, g):
        return self.rows.setdefault(g, {'actor_frames': 0, 'actors': set(), 'switches': 0,
                                        'pos': [], 'v_static': [], 'v_moving': [], 'speed_moving': [],
                                        'heading_moving': []})

    def add(self, tracks, truth):
        """tracks: [dict(id, class, xy, v)] (confirmed or coasting);
        truth: actor id -> dict(class, xy, v)."""
        self.frames += 1
        for tr, a in match_by_position(tracks, truth):
            gt = truth[a]
            row = self._row(class_group(gt['class']))
            row['actor_frames'] += 1
            row['actors'].add(a)
            prev = self.last_track.get(a)
            if prev is not None and prev != tr['id']:
                row['switches'] += 1
            self.last_track[a] = tr['id']
            self.track_actors.setdefault(tr['id'], set()).add(a)

            row['pos'].append(float(np.hypot(*(np.asarray(tr['xy']) - gt['xy']))))
            v, vg = np.asarray(tr['v'], float), np.asarray(gt['v'], float)
            sg = float(np.linalg.norm(vg))
            err = float(np.linalg.norm(v - vg))
            if sg < 0.2:
                row['v_static'].append(err)
            else:
                row['v_moving'].append(err)
                row['speed_moving'].append(abs(float(np.linalg.norm(v)) - sg))
                if sg > 0.5 and np.linalg.norm(v) > 0.1:
                    d = math.atan2(v[1], v[0]) - math.atan2(vg[1], vg[0])
                    row['heading_moving'].append(abs(math.degrees((d + math.pi) % (2 * math.pi) - math.pi)))

    def summary(self):
        swapped = sum(1 for s in self.track_actors.values() if len(s) > 1)

        def m(x):
            return float(np.mean(x)) if len(x) else float('nan')

        def p95(x):
            return float(np.percentile(x, 95)) if len(x) else float('nan')

        out = {'frames': self.frames, 'tracks': len(self.track_actors), 'swapped_tracks': swapped, 'groups': {}}
        for g, r in sorted(self.rows.items()):
            out['groups'][g] = {
                'actors': len(r['actors']), 'actor_frames': r['actor_frames'], 'id_switches': r['switches'],
                'pos_mae': m(r['pos']), 'v_static_mae': m(r['v_static']), 'n_static': len(r['v_static']),
                'v_moving_mae': m(r['v_moving']), 'v_moving_p95': p95(r['v_moving']),
                'n_moving': len(r['v_moving']), 'speed_moving_mae': m(r['speed_moving']),
                'heading_moving_mae': m(r['heading_moving'])}
        return out


def format_summary(s, title):
    lines = [title + ' -- %d frames, %d track ids, %d tracks carried more than one actor'
             % (s['frames'], s['tracks'], s['swapped_tracks']),
             '  class       actors  actor-frames  id switches | pos MAE | velocity error MAE: static (n)'
             '      moving (n)        p95 | speed MAE  heading MAE (moving)']
    for g, r in s['groups'].items():
        lines.append('  %-11s %6d  %12d  %11d | %5.2f m | %5.2f m/s (%5d)  %5.2f m/s (%5d)  %5.2f | %5.2f m/s  %5.1f deg'
                     % (g, r['actors'], r['actor_frames'], r['id_switches'], r['pos_mae'], r['v_static_mae'],
                        r['n_static'], r['v_moving_mae'], r['n_moving'], r['v_moving_p95'],
                        r['speed_moving_mae'], r['heading_moving_mae']))
    return '\n'.join(lines)
