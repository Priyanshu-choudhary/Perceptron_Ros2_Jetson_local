"""Tests for the lidar preprocessing scores (cloud_eval.py)."""

import numpy as np

from perceptron_perception import cloud_eval as ce


def test_match_labels_uses_nearest_within_tolerance():
    gt = np.array([[0, 0, 0], [5, 0, 0]], dtype=float)
    idx, tag = np.array([0, 7], np.uint32), np.array([1, 14], np.uint32)
    q = np.array([[0.02, 0, 0], [5.03, 0, 0], [2.5, 0, 0]])
    i, t, ok = ce.match_labels(q, gt, idx, tag, match_tol=0.1)
    assert ok.tolist() == [True, True, False]
    assert i.tolist()[:2] == [0, 7] and t.tolist()[:2] == [1, 14]


def test_scores_ground_self_and_clusters():
    s = ce.FrameScores(min_hits=3)
    # ground cloud: 4 road points, 1 car point wrongly called ground
    g_lab = (np.zeros(5, np.uint32), np.array([1, 1, 1, 1, 14], np.uint32), np.ones(5, bool))
    # obstacles: car 7 (5 pts, cluster 0), person 9 (4 pts: 2 in cluster 1, 2 in cluster 2),
    # one road point missed by the ground filter
    o_idx = np.array([7] * 5 + [9] * 4 + [0], np.uint32)
    o_tag = np.array([14] * 5 + [12] * 4 + [1], np.uint32)
    o_lab = (o_idx, o_tag, np.ones(10, bool))
    cl = np.array([0] * 5 + [1, 1, 2, 2] + [-1])
    xy = np.array([[8.0, 0]] * 5 + [[15.0, 1]] * 4 + [[3.0, 0]])
    raw_idx = np.array([7, 9, 0, 99, 99], np.uint32)            # 99 = host vehicle
    s.add(g_lab, o_lab, cl, xy, raw_idx, ego_id=99)
    out = s.summary()
    assert abs(out['ground_precision'] - 4 / 5) < 1e-9
    assert abs(out['ground_recall'] - 4 / 5) < 1e-9
    assert out['self_raw_pct'] == 40.0 and out['self_kept_pct'] == 0.0
    assert out['actors']['vehicle 0-10 m'] == (1, 100.0, 0.0, 5.0)
    assert out['actors']['pedestrian 10-20 m'] == (1, 100.0, 100.0, 4.0)   # matched but split in two
