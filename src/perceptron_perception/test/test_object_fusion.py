"""Tests for camera-lidar object fusion and its scoring.

test_pedestrian_in_front_of_car_goes_to_its_own_box is the case the cluster +
horizontal-overlap rule exists for: the pedestrian is inside the car's box too.
"""

import math

import numpy as np

from perceptron_perception import fusion_eval as fe
from perceptron_perception import object_fusion as of

F, W, H = 400.0, 800, 400
K = np.array([[F, 0, (W - 1) / 2], [0, F, (H - 1) / 2], [0, 0, 1.0]])
NO_DIST = np.zeros(5)
# base_footprint (x fwd, y left, z up) -> optical (x right, y down, z fwd), camera at the origin
OPT_FROM_BASE = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)


def strip(x, y_left, y_right, z0=0.1, z1=0.4, step=0.05):
    """Lidar-like points on the near face of an object: a low strip at depth x."""
    ys = np.arange(y_right, y_left, step)
    zs = np.arange(z0, z1, step)
    yy, zz = np.meshgrid(ys, zs)
    return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])


def box_px(x, y_left, y_right, z_bot, z_top):
    """Pixel box of an object spanning [y_right, y_left] x [z_bot, z_top] at depth x."""
    u = [(W - 1) / 2 - F * y / x for y in (y_left, y_right)]
    v = [(H - 1) / 2 - F * z / x for z in (z_top, z_bot)]
    return (min(u), min(v), max(u), max(v))


def run(points, labels, boxes):
    uv, depth, idx = of.project(points, OPT_FROM_BASE, K, NO_DIST, W, H, min_z=0.1, max_radius=3.0)
    return of.associate(boxes, uv, depth, idx, labels)


def test_pedestrian_in_front_of_car_goes_to_its_own_box():
    car = strip(15.0, 2.0, -2.0)                  # 4 m wide car face at 15 m
    ped = strip(10.0, 0.25, -0.25)                # person at 10 m, in front of the car
    pts = np.vstack([car, ped])
    labels = np.r_[np.zeros(len(car), int), np.ones(len(ped), int)]
    boxes = [box_px(15.0, 2.0, -2.0, 0.0, 1.5), box_px(10.0, 0.3, -0.3, 0.0, 1.8)]
    got, how = run(pts, labels, boxes)
    assert set(got[0]) == set(range(len(car)))
    assert set(got[1]) == set(range(len(car), len(pts)))
    assert how == ['cluster', 'cluster']


def test_undetected_pedestrian_in_front_does_not_join_the_car():
    car = strip(15.0, 2.0, -2.0)
    ped = strip(10.0, 0.25, -0.25)                # no box of its own (detector missed it)
    pts = np.vstack([car, ped])
    labels = np.r_[np.zeros(len(car), int), np.ones(len(ped), int)]
    got, _ = run(pts, labels, [box_px(15.0, 2.0, -2.0, 0.0, 1.5)])
    assert set(got[0]) == set(range(len(car)))


def test_car_split_into_two_clusters_is_rejoined():
    front = strip(12.0, 1.0, -1.0)
    side = strip(13.5, -1.0, -1.1)                 # side face a little further, own cluster
    pts = np.vstack([front, side])
    labels = np.r_[np.zeros(len(front), int), np.ones(len(side), int)]
    got, _ = run(pts, labels, [box_px(12.0, 1.0, -1.2, 0.0, 1.5)])
    assert len(got[0]) == len(pts)


def test_wall_behind_does_not_join():
    obj = strip(10.0, 0.5, -0.5)
    wall = strip(25.0, 15.0, -15.0)               # wide wall, mostly outside the box
    pts = np.vstack([obj, wall])
    labels = np.r_[np.zeros(len(obj), int), np.ones(len(wall), int)]
    got, _ = run(pts, labels, [box_px(10.0, 0.55, -0.55, 0.0, 1.0)])
    assert set(got[0]) == set(range(len(obj)))


def test_unclustered_far_object_uses_nearest_depth_group():
    obj = strip(30.0, 0.2, -0.2, step=0.1)        # too sparse to have been clustered
    behind = strip(45.0, 0.3, -0.3, step=0.1)
    pts = np.vstack([obj, behind])
    labels = -np.ones(len(pts), int)
    got, how = run(pts, labels, [box_px(30.0, 0.4, -0.4, 0.0, 1.8)])
    assert set(got[0]) == set(range(len(obj))) and how == ['loose points']


def test_describe_range_is_nearest_surface():
    pts = np.vstack([strip(10.0, 0.5, -0.5), strip(12.0, 0.5, -0.5)])
    d = of.describe(pts, np.zeros(3))
    assert 9.99 < d['range'] < 10.1
    assert 10.9 < d['centroid'][0] < 11.1


def test_closest_point_distance_on_rotated_box():
    q = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))     # 45 deg about z
    d = fe.closest_point_distance((0, 0, 0), (10, 0, 0), q, (2.0, 2.0, 2.0))
    assert abs(d - (10 - math.sqrt(2))) < 1e-9
    assert fe.closest_point_distance((10, 0, 0), (10, 0, 0), q, (2, 2, 2)) == 0.0


def test_fusion_scores():
    s = fe.FusionScores()
    boxes = [('1', 'pedestrian', True), ('2', 'car', True), ('3', 'car', False)]
    objects = {'1': {'range': 10.2, 'xy': (10.3, 0.0), 'purity': 1.0},
               '2': {'range': 14.0, 'xy': (15.0, 0.0), 'purity': 0.2}}
    truth = {'1': {'range': 10.0, 'xy': (10.0, 0.0)}, '2': {'range': 15.0, 'xy': (17.0, 0.0)},
             '3': {'range': 16.0, 'xy': (18.0, 0.0)}}
    s.add(boxes, objects, truth)
    out = s.summary()
    ped, car = out['pedestrian 10-20 m'], out['vehicle 10-20 m']
    assert ped['coverage_pct'] == 100.0 and ped['correct_pct'] == 100.0
    assert abs(ped['range_bias'] - 0.2) < 1e-9
    assert car['coverage_pct'] == 50.0 and car['no_points_pct'] == 50.0 and car['correct_pct'] == 0.0
