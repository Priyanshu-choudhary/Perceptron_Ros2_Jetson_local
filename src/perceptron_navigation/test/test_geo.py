"""Tests for GPS conversion and route sub-goals (geo.py)."""

import math

import pytest

from perceptron_navigation import geo

LAT0, LON0 = 19.1334, 72.9133


def test_enu_round_trip_and_bridge_convention():
    lat, lon = geo.enu_to_latlon(250.0, -120.0, LAT0, LON0)
    e, n = geo.latlon_to_enu(lat, lon, LAT0, LON0)
    assert e == pytest.approx(250.0, abs=1e-6) and n == pytest.approx(-120.0, abs=1e-6)
    # perceptron_carla.sim_sensors.offset_latlon: dlat = n / 111320, dlon = e / (111320 cos lat0)
    assert lat == pytest.approx(LAT0 - 120.0 / 111320.0)
    assert lon == pytest.approx(LON0 + 250.0 / (111320.0 * math.cos(math.radians(LAT0))))


def test_route_hands_out_carrots_then_the_goal():
    r = geo.Route([(0, 0), (100, 0)], spacing=30.0, advance=12.0)
    assert r.next_subgoal((0, 0)) == pytest.approx((30.0, 0.0, 0.0))
    assert r.next_subgoal((10, 0)) is None                       # 20 m from the carrot: keep it
    assert r.next_subgoal((20, 1)) == pytest.approx((50.0, 0.0, 0.0))
    assert r.next_subgoal((45, 0)) == pytest.approx((75.0, 0.0, 0.0)) and not r.is_final()
    g = r.next_subgoal((65, 0))                                  # 95 m would leave a 5 m stub: goal
    assert g == pytest.approx((100.0, 0.0, 0.0)) and r.is_final()
    assert r.next_subgoal((95, 0)) is None


def test_route_progress_and_heading_on_a_bend():
    r = geo.Route([(0, 0), (10, 0), (10, 10)], spacing=5.0)
    assert r.length == pytest.approx(20.0)
    assert r.progress((12, 4)) == pytest.approx(14.0)
    x, y, yaw = r.at(15.0)
    assert (x, y) == pytest.approx((10.0, 5.0)) and yaw == pytest.approx(math.pi / 2)


def test_short_route_is_a_single_goal():
    r = geo.Route([(0, 0), (8, 6)], spacing=30.0)
    assert r.next_subgoal((0, 0)) == pytest.approx((8.0, 6.0, math.atan2(6, 8))) and r.is_final()


def test_road_graph_routes_round_the_corner_not_through_the_block():
    # an L-shaped street: east along y = 0, a junction lane, then north along x = 50
    east = [(float(x), 0.0) for x in range(0, 46, 3)]
    turn = [(45.0, 0.0), (48.5, 1.5), (50.0, 5.0)]
    north = [(50.0, float(y)) for y in range(5, 60, 3)]
    g = geo.RoadGraph([east, turn, north])
    r = g.route((1.0, 0.5), (50.5, 56.0))
    assert r[0] == (1.0, 0.5) and r[-1] == (50.5, 56.0)
    length = sum(math.dist(a, b) for a, b in zip(r, r[1:]))
    assert 95 < length < 110                      # along the streets (a straight line is ~74 m)
    assert all(not (5 < x < 45 and 5 < y < 50) for x, y in r)   # never inside the block


def test_road_graph_is_undirected_and_reports_disconnection():
    a = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    b = [(100.0, 0.0), (110.0, 0.0)]
    g = geo.RoadGraph([a, b])
    back = g.route((19.0, 0.0), (1.0, 0.0))       # against the drawing direction: allowed
    assert back[1] == (20.0, 0.0) and back[-2] == (0.0, 0.0)
    assert g.route((0.0, 0.0), (110.0, 0.0)) is None


def test_lane_change_edges_avoid_the_junction_detour():
    # two parallel lanes 3.5 m apart joined only at the far end (x = 100): starting on the
    # lower one, a goal on the upper one 20 m along must not go round via x = 100
    lower = [(float(x), 0.0) for x in range(0, 101, 2)]
    upper = [(float(x), 3.5) for x in range(0, 101, 2)]
    link = [(100.0, 0.0), (101.5, 1.75), (100.0, 3.5)]
    r = geo.RoadGraph([lower, link, upper]).route((0.0, 0.0), (20.0, 3.5))
    assert max(x for x, _ in r) < 25
    r = geo.RoadGraph([lower, link, upper], lane_change=0).route((0.0, 0.0), (20.0, 3.5))
    assert max(x for x, _ in r) > 99                 # without lane changes: the long way round
