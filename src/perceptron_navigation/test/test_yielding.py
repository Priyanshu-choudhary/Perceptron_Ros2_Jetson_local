"""Tests for the predictive speed limit (yielding.py)."""

import math

import numpy as np
import pytest

from perceptron_navigation import yielding as y

NOSE, HALF = 2.1, 0.8


def walker(x0, y0, vx, vy, dt=0.1, n=21, r=0.4):
    return {'path': np.array([(x0 + vx * k * dt, y0 + vy * k * dt) for k in range(n)]), 'path_dt': dt, 'radius': r}


def test_corridor_locate_on_a_bend():
    c = y.Corridor([(0, 0), (10, 0), (10, 10)])
    s, off = c.locate((11, 4))
    assert s == pytest.approx(14.0) and off == pytest.approx(1.0)


def test_person_crossing_ahead_slows_or_stops_the_robot():
    c = y.Corridor.straight(0, 0, 0)
    # 8 m ahead, 3 m to the left, walking right at 1.2 m/s: in the corridor from ~1 s to ~3.8 s
    cap, why = y.speed_cap(c, [walker(8.0, 3.0, 0.0, -1.2)], HALF, NOSE)
    room = 8.0 - NOSE - 1.5
    assert why[0] == pytest.approx(3.9)            # extended past the 2 s prediction: out at 3.92 s
    assert cap == pytest.approx(room / 3.9)
    cap2, why2 = y.speed_cap(c, [walker(8.0, 3.0, 0.0, -1.2)], HALF, NOSE, extend=0.0)
    assert why2[0] == pytest.approx(2.0) and cap2 > cap   # 2 s alone lets the robot in too early


def test_person_standing_in_the_corridor_means_creep_up_and_stop_short():
    c = y.Corridor.straight(0, 0, 0)
    cap, _ = y.speed_cap(c, [walker(6.0, 0.2, 0.0, 0.0)], HALF, NOSE)
    assert cap == pytest.approx((6.0 - NOSE - 1.5) / 6.0)        # still there at the 6 s extension
    cap, _ = y.speed_cap(c, [walker(6.0, 0.2, 0.0, 0.0, n=1)], HALF, NOSE)   # no prediction: braking bound
    assert cap == pytest.approx(math.sqrt(2 * 2.0 * (6.0 - NOSE - 1.5)))
    cap, _ = y.speed_cap(c, [walker(3.0, 0.0, 0.0, 0.0)], HALF, NOSE)   # within the gap: stop now
    assert cap == 0.0


def test_people_beside_behind_or_walking_away_do_not_limit():
    c = y.Corridor.straight(0, 0, 0)
    assert y.speed_cap(c, [walker(5.0, 3.5, 1.0, 0.0)], HALF, NOSE)[0] is None    # parallel on the sidewalk
    assert y.speed_cap(c, [walker(-4.0, 0.0, -1.0, 0.0)], HALF, NOSE)[0] is None  # behind
    assert y.speed_cap(c, [walker(1.0, 1.2, 0.0, 1.0)], HALF, NOSE)[0] is None    # beside the body, leaving


def test_limiter_drops_at_once_and_rises_slowly():
    lim = y.Limiter(rise=1.0, release=3.0)
    assert lim.update(None, 0.1) is None
    assert lim.update(0.5, 0.1) == 0.5
    assert lim.update(2.5, 0.1) == pytest.approx(0.6)
    for _ in range(40):
        v = lim.update(None, 0.1)
    assert v is None                                  # back above the release speed: no limit
