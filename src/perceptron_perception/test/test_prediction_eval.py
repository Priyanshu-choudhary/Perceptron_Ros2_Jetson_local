"""Tests for Stage 7 prediction scoring (prediction_eval.py)."""

import numpy as np

from perceptron_perception import prediction_eval as pe


def test_scores_error_baselines_and_coverage():
    s = pe.PredictionScores([0.5, 1.0])
    # a walker at (0,0) moving +x at 1 m/s; ours predicts it perfectly, with 0.1 m sigma
    s.add_predictions(10.0, [{'actor': 7, 'cls': 'pedestrian', 'moving': True, 'model': 1,
                              'means': np.array([[0.5, 0.0], [1.0, 0.0]]),
                              'covs': np.array([np.eye(2) * 0.01] * 2),
                              'pos': np.array([0.0, 0.0]), 'vel': np.array([1.2, 0.0])}])
    s.add_truth(10.5, {7: np.array([0.5, 0.0])})
    s.add_truth(11.0, {7: np.array([1.0, 0.3])})          # 0.3 m off at 1 s: outside 95 %
    out = s.summary()[('pedestrian', 'moving')]
    h05, h1 = out
    assert h05['ours'] == 0.0 and h05['stay'] == 0.5 and abs(h05['raw'] - 0.1) < 1e-9
    assert h05['in95'] == 100.0
    assert abs(h1['ours'] - 0.3) < 1e-9 and h1['in95'] == 0.0
    assert not s.pending


def test_missing_actor_is_dropped_not_scored():
    s = pe.PredictionScores([0.5])
    s.add_predictions(0.0, [{'actor': 1, 'cls': 'car', 'moving': False, 'model': 0,
                             'means': np.zeros((1, 2)), 'covs': np.array([np.eye(2)]),
                             'pos': np.zeros(2), 'vel': np.zeros(2)}])
    s.add_truth(0.5, {})
    assert not s.summary() and not s.pending
