"""Scoring lidar preprocessing against CARLA's semantic lidar (no ROS).

The semantic lidar is a twin of the real one: same pose, same rays, and every
return carries the semantic tag and the actor id it hit. Each processed point
(ground or obstacle) is matched to its nearest twin point; anything further
than match_tol away (voxel jitter, range noise) stays unlabelled. From those
labels:

  ground      precision / recall of the ground class against the ground tags
  self        share of points on the host vehicle, before and after the self box
  clusters    per actor with at least min_hits obstacle points:
                matched     one cluster holds >= 50 % of its points AND >= 50 % of
                            that cluster is this actor
                over-split  its points are spread over >= 2 clusters with >= 20 % each
              per cluster: under-split when >= 2 actors each own >= 20 % of it
"""

import numpy as np
from scipy.spatial import cKDTree

# CARLA 0.9.15 tags a rover can drive on: road, sidewalk, terrain, road_line, ground, rail_track
GROUND_TAGS = np.array([1, 2, 10, 24, 25, 27], dtype=np.uint32)
TAG_NAMES = {0: 'unlabeled', 1: 'road', 2: 'sidewalk', 3: 'building', 4: 'wall', 5: 'fence', 6: 'pole',
             7: 'traffic_light', 8: 'traffic_sign', 9: 'vegetation', 10: 'terrain', 11: 'sky',
             12: 'pedestrian', 13: 'rider', 14: 'car', 15: 'truck', 16: 'bus', 17: 'train',
             18: 'motorcycle', 19: 'bicycle', 20: 'static', 21: 'dynamic', 22: 'other', 23: 'water',
             24: 'road_line', 25: 'ground', 26: 'bridge', 27: 'rail_track', 28: 'guard_rail'}
PEDESTRIAN_TAGS = (12, 13)
VEHICLE_TAGS = (14, 15, 16, 17, 18, 19)
RANGE_BINS = (0.0, 10.0, 20.0, 40.0)


def match_labels(points, gt_xyz, gt_idx, gt_tag, match_tol=0.1):
    """(idx, tag, matched) for each processed point from its nearest ground-truth point."""
    if len(points) == 0 or len(gt_xyz) == 0:
        z = np.zeros(len(points), dtype=np.uint32)
        return z, z, np.zeros(len(points), dtype=bool)
    dist, nn = cKDTree(gt_xyz).query(points, distance_upper_bound=match_tol)
    ok = np.isfinite(dist)
    nn = np.where(ok, nn, 0)
    return np.where(ok, gt_idx[nn], 0), np.where(ok, gt_tag[nn], 0), ok


def class_of(tags):
    vals, counts = np.unique(tags, return_counts=True)
    t = int(vals[np.argmax(counts)])
    if t in PEDESTRIAN_TAGS:
        return 'pedestrian'
    if t in VEHICLE_TAGS:
        return 'vehicle'
    return 'other'


class FrameScores:
    """Accumulates the scores of many frames."""

    def __init__(self, min_hits=10):
        self.min_hits = min_hits
        self.tp = self.fp = self.fn = 0
        self.fp_tags = {}         # what the points wrongly called ground really were
        self.self_raw = self.raw = self.self_kept = self.kept = 0
        self.actors = {}          # (class, bin) -> [count, matched, over_split]
        self.clusters = self.under_split = self.frames = 0

    def add(self, ground_lab, obst_lab, obst_cluster, obst_xy, raw_idx, ego_id):
        """ground_lab / obst_lab: (idx, tag, matched) from match_labels."""
        self.frames += 1
        g_tag, g_ok = ground_lab[1], ground_lab[2]
        o_idx, o_tag, o_ok = obst_lab
        g_is_ground = np.isin(g_tag, GROUND_TAGS)
        o_is_ground = np.isin(o_tag, GROUND_TAGS)
        self.tp += int((g_ok & g_is_ground).sum())
        self.fp += int((g_ok & ~g_is_ground).sum())
        for t, c in zip(*np.unique(g_tag[g_ok & ~g_is_ground], return_counts=True)):
            self.fp_tags[int(t)] = self.fp_tags.get(int(t), 0) + int(c)
        self.fn += int((o_ok & o_is_ground).sum())

        if ego_id is not None:
            self.raw += len(raw_idx)
            self.self_raw += int((raw_idx == ego_id).sum())
            kept_idx = np.r_[ground_lab[0], o_idx]
            self.kept += len(kept_idx)
            self.self_kept += int((kept_idx == ego_id).sum())

        actor = o_ok & (o_idx > 0) & (o_idx != (ego_id if ego_id is not None else -1))
        ids = np.unique(o_idx[actor])
        for a in ids:
            sel = actor & (o_idx == a)
            n = int(sel.sum())
            if n < self.min_hits:
                continue
            cl = obst_cluster[sel]
            r = float(np.linalg.norm(obst_xy[sel].mean(axis=0)))
            b = int(np.searchsorted(RANGE_BINS, r, side='right') - 1)
            if b < 0 or b >= len(RANGE_BINS) - 1:
                continue
            key = (class_of(o_tag[sel]), b)
            rec = self.actors.setdefault(key, [0, 0, 0, 0])
            rec[0] += 1
            rec[3] += n
            valid = cl[cl >= 0]
            if len(valid):
                vals, counts = np.unique(valid, return_counts=True)
                c = vals[np.argmax(counts)]
                coverage = counts.max() / n
                in_c = obst_cluster == c
                purity = (o_idx[in_c] == a).sum() / in_c.sum()
                if coverage >= 0.5 and purity >= 0.5:
                    rec[1] += 1
                if (counts >= 0.2 * n).sum() >= 2:
                    rec[2] += 1

        for c in np.unique(obst_cluster[obst_cluster >= 0]):
            self.clusters += 1
            in_c = (obst_cluster == c) & actor
            if in_c.sum() == 0:
                continue
            vals, counts = np.unique(o_idx[in_c], return_counts=True)
            if (counts >= 0.2 * (obst_cluster == c).sum()).sum() >= 2:
                self.under_split += 1

    def summary(self):
        s = {'frames': self.frames,
             'ground_precision': self.tp / max(self.tp + self.fp, 1),
             'ground_recall': self.tp / max(self.tp + self.fn, 1),
             'self_raw_pct': 100.0 * self.self_raw / max(self.raw, 1),
             'self_kept_pct': 100.0 * self.self_kept / max(self.kept, 1),
             'clusters_per_frame': self.clusters / max(self.frames, 1),
             'under_split_pct': 100.0 * self.under_split / max(self.clusters, 1),
             'ground_fp_by_class': sorted(((TAG_NAMES.get(t, str(t)), 100.0 * c / max(self.fp, 1))
                                           for t, c in self.fp_tags.items()), key=lambda kv: -kv[1])[:4],
             'actors': {}}
        for (cls, b), (n, m, o, pts) in sorted(self.actors.items()):
            s['actors']['%s %g-%g m' % (cls, RANGE_BINS[b], RANGE_BINS[b + 1])] = (
                n, 100.0 * m / n, 100.0 * o / n, pts / n)
        return s


def format_summary(s, title):
    lines = [title,
             '  ground   precision %.3f  recall %.3f' % (s['ground_precision'], s['ground_recall']),
             '           wrongly called ground: ' + ', '.join('%s %.0f %%' % kv for kv in s['ground_fp_by_class']),
             '  self     host-vehicle points: %.2f %% of raw, %.2f %% after the self box'
             % (s['self_raw_pct'], s['self_kept_pct']),
             '  clusters %.1f per frame, %.1f %% merge two or more actors'
             % (s['clusters_per_frame'], s['under_split_pct']),
             '  actor recall (actor-frames with >= min_hits obstacle points):']
    for k, (n, m, o, pts) in s['actors'].items():
        lines.append('    %-22s %6d   matched %5.1f %%   split %5.1f %%   %5.1f pts each' % (k, n, m, o, pts))
    return '\n'.join(lines)
