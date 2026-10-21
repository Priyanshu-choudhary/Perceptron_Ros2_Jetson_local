"""Tests for Stage 7 prediction (prediction.py): one per constraint in its docstring."""

import math

import numpy as np

from perceptron_perception import prediction as pr

H = [0.5, 1.0, 2.0]


def P(pos_var=0.01, vel_var=1e-4):
    return np.diag([pos_var, pos_var, vel_var, vel_var])


def test_static_gate_keeps_object_in_place():
    out = pr.predict([5.0, 2.0, 0.4, 0.0], P(), 'car', dynamic=False, horizons=H)
    assert out['model'] == pr.STATIC
    assert np.allclose(out['means'], [[5.0, 2.0]] * 3)
    assert np.allclose(out['covs'][2], np.diag([0.01 + 1.44, 0.01 + 1.44]))      # + car pos_floor 1.2^2


def test_constant_velocity_mean():
    out = pr.predict([0.0, 0.0, 1.2, 0.0], P(), 'pedestrian', dynamic=True, horizons=H)
    assert out['model'] == pr.CV
    assert np.allclose(out['means'][:, 0], [0.6, 1.2, 2.4], atol=1e-3)


def test_shrinkage_pulls_noisy_slow_velocity_towards_zero():
    v = pr.shrink_velocity(np.array([0.3, 0.0]), np.eye(2) * 0.09)
    assert math.isclose(v[0], 0.15)
    v = pr.shrink_velocity(np.array([10.0, 0.0]), np.eye(2) * 0.09)
    assert v[0] > 9.99


def test_speed_cap():
    out = pr.predict([0.0, 0.0, 6.0, 0.0], P(), 'pedestrian', dynamic=True, horizons=H)
    assert math.isclose(out['speed'], 3.0)
    assert math.isclose(out['means'][2][0], 6.0, rel_tol=1e-6)


def test_ctrv_follows_the_circle():
    v, w = 8.0, 0.2
    out = pr.predict([0.0, 0.0, v, 0.0], P(), 'car', dynamic=True, horizons=H, yaw=(w, 0.01))
    assert out['model'] == pr.CTRV
    r = out['speed'] / w                                  # speed after shrinkage, ~8.0
    centre = np.array([0.0, r])                           # turning left about (0, ~40)
    for m in out['means']:
        assert abs(np.linalg.norm(m - centre) - r) < 1e-6
    assert out['means'][2][1] > 0.5                       # it has turned left


def test_yaw_rate_capped_by_lateral_grip():
    out = pr.predict([0.0, 0.0, 8.0, 0.0], P(), 'car', dynamic=True, horizons=H, yaw=(1.0, 0.01))
    assert math.isclose(out['yaw_rate'], 4.0 / out['speed'])   # a_lat_max / v


def test_unmeasured_turn_stays_cv():
    out = pr.predict([0.0, 0.0, 8.0, 0.0], P(), 'car', dynamic=True, horizons=H, yaw=(0.1, 0.2))
    assert out['model'] == pr.CV


def test_uncertainty_grows_with_velocity_variance():
    small = pr.predict([0.0, 0.0, 2.0, 0.0], P(vel_var=0.01), 'pedestrian', True, H)['covs']
    big = pr.predict([0.0, 0.0, 2.0, 0.0], P(vel_var=0.09), 'pedestrian', True, H)['covs']
    assert all(big[i][1, 1] > small[i][1, 1] for i in range(3))
    assert small[2][1, 1] > small[0][1, 1]                # and with the horizon
    # +/-0.3 m/s at 2 s alone is 0.6 m sigma: (0.3 * 2)^2 = 0.36 is inside the variance
    assert big[2][1, 1] >= 0.36


def test_covariance_capped_to_reachable_distance():
    out = pr.predict([0.0, 0.0, 1.0, 0.0], P(vel_var=100.0), 'pedestrian', True, H)
    for t, S in zip(H, out['covs']):
        reach = 1.0 * t + 3.0 * t * t / 2 + 2 * (0.01 + 0.15 ** 2) ** 0.5
        assert 2 * math.sqrt(np.linalg.eigvalsh(S).max()) <= reach + 1e-6


def test_yaw_rate_estimator_recovers_turn():
    est = pr.YawRateEstimator(window=1.0)
    for k in range(25):
        t = k * 0.05
        h = 0.2 * t
        est.add(t, 8 * math.cos(h), 8 * math.sin(h))
    w, s = est.estimate()
    assert abs(w - 0.2) < 1e-3 and s < 0.01


def test_motion_gate_hysteresis():
    g = pr.MotionGate(on_frames=3, off_frames=10)
    assert [g.update(True) for _ in range(3)] == [False, False, True]
    assert all(g.update(False) for _ in range(9))
    assert g.update(False) is False
