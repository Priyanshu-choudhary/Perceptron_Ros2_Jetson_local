"""Tests for the simulated-sensor models and the localization metrics.

The conversion tests pin the conventions measured on CARLA 0.9.15 (gyro z
follows CARLA's clockwise yaw, +y is south); if CARLA ever changes them,
these are the tests that should fail rather than the EKF quietly diverging.
"""

import math

import numpy as np

from perceptron_carla import loc_metrics as lm
from perceptron_carla import sim_sensors as ss


def test_turning_left_is_positive_yaw_rate_in_ros():
    # Measured: turning left in CARLA decreases yaw and gives a negative gyro z.
    _, _, wz = ss.gyro_carla_to_ros(0.0, 0.0, -0.3)
    assert wz > 0


def test_rotation_conversion_matches_quaternion_yaw():
    r, p, y = ss.rotation_carla_to_ros_rpy(0.0, 0.0, -90.0)     # CARLA yaw -90 = facing north (-y)
    assert math.isclose(y, math.pi / 2)
    q = ss.quat_from_rpy(r, p, y)
    assert math.isclose(lm.yaw_from_quat(*q), math.pi / 2, abs_tol=1e-12)


def test_accel_gravity_stays_up():
    assert ss.accel_carla_to_ros(0.0, 0.0, 9.81)[2] == 9.81


def test_offset_latlon_one_metre():
    lat, lon = ss.offset_latlon(0.0, 0.0, 1.0, 1.0)
    assert math.isclose(lat * ss.EARTH_M_PER_DEG, 1.0, rel_tol=1e-9)
    assert math.isclose(lon * ss.EARTH_M_PER_DEG, 1.0, rel_tol=1e-9)


def test_wheel_odometry_scale_and_noise():
    rng = np.random.default_rng(0)
    m = ss.WheelOdometry(rng, v_scale_error=0.02, w_scale_error=0.0, v_noise_std=0.02, w_noise_std=0.0)
    v = np.array([m.measure(5.0, 0.0)[0] for _ in range(4000)])
    assert abs(v.mean() - 5.1) < 0.005          # 2 % scale error is a bias, not noise
    assert abs(v.std() - 0.02) < 0.003


def test_imu_yaw_offset_walks_but_stays_bounded():
    rng = np.random.default_rng(1)
    m = ss.ImuOrientation(rng, dt=0.05, yaw_bias=math.radians(2.0), yaw_walk_std=math.radians(0.05),
                          yaw_noise_std=0.0, tilt_noise_std=0.0)
    errs = [m.measure(0.0, 0.0, 0.0)[2] for _ in range(1200)]   # one minute
    assert abs(errs[0] - math.radians(2.0)) < math.radians(0.1)
    assert abs(errs[-1]) < math.radians(5.0)                    # 2 deg + ~0.4 deg/min walk
    assert errs[-1] != errs[0]


def test_gnss_noise_and_bias_in_metres():
    rng = np.random.default_rng(2)
    g = ss.Gnss(rng, bias_std=0.0, noise_std=1.0)
    fixes = np.array([g.measure(45.0, 7.0) for _ in range(3000)])
    north = (fixes[:, 0] - 45.0) * ss.EARTH_M_PER_DEG
    east = (fixes[:, 1] - 7.0) * ss.EARTH_M_PER_DEG * math.cos(math.radians(45.0))
    assert abs(north.std() - 1.0) < 0.05 and abs(east.std() - 1.0) < 0.05
    assert abs(north.mean()) < 0.06


def test_ground_truth_interpolation_wraps_yaw():
    b = lm.GroundTruthBuffer()
    b.add(0.0, 0.0, 0.0, math.pi - 0.1)
    b.add(1.0, 2.0, 0.0, -math.pi + 0.1)
    x, y, yaw = b.at(0.5)
    assert math.isclose(x, 1.0) and math.isclose(abs(yaw), math.pi, abs_tol=1e-9)
    assert b.at(1.5) is None


def test_error_stats_drift():
    s = lm.ErrorStats()
    for i in range(101):
        gt = (float(i), 0.0, 0.0)
        est = (float(i), 0.01 * i, math.radians(1.0))
        s.add(i * 0.1, est, gt)
    out = s.summary()
    assert math.isclose(out['distance_m'], 100.0)
    assert math.isclose(out['pos_final_m'], 1.0)
    assert math.isclose(out['drift_pct'], 1.0)
    assert math.isclose(out['head_rmse_deg'], 1.0, rel_tol=1e-6)
