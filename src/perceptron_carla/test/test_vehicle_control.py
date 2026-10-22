"""Tests for /cmd_vel -> CARLA control (vehicle_control.py)."""

import math

from perceptron_carla import vehicle_control as vc

DT = 0.05


def ctl(**kw):
    return vc.TwistController(wheelbase=1.5, max_wheel_steer=math.radians(70.0), **kw)


def test_left_turn_is_negative_carla_steer_and_matches_curvature():
    c = ctl(steer_rate=100.0)
    out = c.step(2.0, 0.5, 2.0, DT, 0.0)                 # k = 0.25 1/m -> d = atan(0.375)
    assert out.steer < 0
    assert math.isclose(-out.steer * math.radians(70.0), math.atan(1.5 * 0.25), rel_tol=1e-6)


def test_steering_is_limited_and_rate_limited():
    c = ctl(max_steer=math.radians(35.0), steer_rate=1.0)
    out = c.step(1.0, 5.0, 1.0, DT, 0.0)
    assert math.isclose(out.steer, -0.05)                # 1.0 / s for 0.05 s
    for _ in range(200):
        out = c.step(1.0, 5.0, 1.0, DT, 0.0)
    assert math.isclose(-out.steer * math.radians(70.0), math.radians(35.0), rel_tol=1e-6)
    assert math.isclose(c.min_turning_radius(), 1.5 / math.tan(math.radians(35.0)))


def test_speed_loop_accelerates_cruises_and_brakes():
    c = ctl()
    assert c.step(3.0, 0.0, 0.0, DT, 0.0).throttle > 0.5             # far below: strong throttle
    cruise = c.step(3.0, 0.0, 3.0, DT, 0.0)
    assert 0.0 < cruise.throttle < 0.5 and cruise.brake == 0.0
    fast = c.step(3.0, 0.0, 5.0, DT, 0.0)
    assert fast.brake > 0.0 and fast.throttle == 0.0
    assert c.step(10.0, 0.0, 4.0, DT, 0.0).throttle > 0.0             # clamped to max_speed 4: still below


def test_stale_command_and_estop_brake():
    c = ctl(timeout=0.5)
    out = c.step(3.0, 0.0, 3.0, DT, cmd_age=0.6)
    assert out.state == 'stale' and out.brake == 1.0 and out.throttle == 0.0
    out = c.step(3.0, 0.0, 0.05, DT, cmd_age=0.0, estop=True)
    assert out.state == 'estop' and out.brake > 0.5


def test_reverse_waits_for_a_stop_then_backs_up():
    c = ctl()
    out = c.step(-0.5, 0.0, 2.0, DT, 0.0)                # still rolling forward
    assert out.brake == 1.0 and not out.reverse and out.state == 'changing direction'
    out = c.step(-0.5, 0.0, 0.0, DT, 0.0)
    assert out.reverse and out.throttle > 0.0
    out = c.step(-0.5, 0.3, -0.5, DT, 0.0)               # backing while yawing left: wheels turn right
    assert out.steer > 0
