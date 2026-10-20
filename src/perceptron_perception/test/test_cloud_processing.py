"""Tests for lidar preprocessing on synthetic scenes.

test_sidewalk_is_ground_but_car_is_not is the one the design hinges on: a
single plane would call a raised sidewalk an obstacle, a pure height grid would
call a car's underside ground. See cloud_processing.py.
"""

import math

import numpy as np

from perceptron_perception import cloud_processing as cpr


def plane_points(x0, x1, y0, y1, z, step=0.1, rng=None, noise=0.01):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    p = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])
    if rng is not None:
        p[:, 2] += rng.normal(0, noise, len(p))
    return p


def box_surface(cx, cy, sx, sy, z0, z1, step=0.08):
    """Points on the four vertical sides of a box, like a lidar sees them."""
    pts = []
    for z in np.arange(z0, z1, step):
        for x in np.arange(cx - sx / 2, cx + sx / 2, step):
            pts += [(x, cy - sy / 2, z), (x, cy + sy / 2, z)]
        for y in np.arange(cy - sy / 2, cy + sy / 2, step):
            pts += [(cx - sx / 2, y, z), (cx + sx / 2, y, z)]
    return np.array(pts)


def test_range_and_box_masks():
    p = np.array([[0.2, 0, 0], [5.0, 0, 0], [200.0, 0, 0]])
    assert cpr.range_mask(p, 0.5, 100).tolist() == [False, True, False]
    assert cpr.box_mask(p, [0, 1, -1, 1, -1, 1]).tolist() == [True, False, False]


def test_voxel_downsample_keeps_one_per_voxel():
    p = np.array([[0.01, 0.01, 0.0], [0.02, 0.03, 0.0], [0.5, 0.0, 0.0]])
    assert len(cpr.voxel_downsample(p, 0.1)) == 2
    assert len(cpr.voxel_downsample(p, 0.0)) == 3


def test_ransac_recovers_a_tilted_plane():
    rng = np.random.default_rng(0)
    p = plane_points(-10, 10, -10, 10, 0.0, step=0.3, rng=rng)
    tilt = math.radians(3.0)
    p[:, 2] += p[:, 0] * math.tan(tilt) + 0.05            # 3 deg slope, 5 cm offset
    n, d = cpr.fit_plane_ransac(p, threshold=0.05, rng=rng)
    assert abs(cpr.plane_tilt_deg(n) - 3.0) < 0.2
    assert abs(cpr.plane_height_at_origin((n, d)) - 0.05) < 0.02


def test_sidewalk_is_ground_but_car_is_not():
    rng = np.random.default_rng(1)
    road = plane_points(0, 30, -4, 3, 0.0, rng=rng)
    sidewalk = plane_points(0, 30, 3, 6, 0.15, rng=rng)          # 15 cm curb
    car = box_surface(12.0, 0.0, 4.5, 1.8, 0.3, 1.5)              # body starts 30 cm up
    person = box_surface(20.0, 4.5, 0.5, 0.4, 0.15, 1.8)          # standing on the sidewalk
    xyz = np.vstack([road, sidewalk, car, person])
    truth = np.r_[np.ones(len(road) + len(sidewalk), bool), np.zeros(len(car) + len(person), bool)]
    ground, plane = cpr.segment_ground(xyz, rng=rng)
    assert abs(cpr.plane_height_at_origin(plane)) < 0.03 and cpr.plane_tilt_deg(plane[0]) < 0.5
    recall = (ground & truth).sum() / truth.sum()
    obstacle_kept = (~ground & ~truth).sum() / (~truth).sum()
    assert recall > 0.99, recall
    assert obstacle_kept > 0.9, obstacle_kept


def test_two_people_close_together_stay_two_clusters():
    a = box_surface(10.0, 0.0, 0.5, 0.4, 0.2, 1.8)
    b = box_surface(10.0, 1.2, 0.5, 0.4, 0.2, 1.8)       # 0.8 m gap between them
    car = box_surface(20.0, -3.0, 4.5, 1.8, 0.3, 1.5)
    xyz = np.vstack([a, b, car])
    labels = cpr.cluster_2d(xyz[:, :2], tolerance=0.5, min_points=5)
    assert len(set(labels[:len(a)])) == 1 and len(set(labels[len(a):len(a) + len(b)])) == 1
    assert labels[0] != labels[len(a)]
    assert labels[-1] == 0                                # biggest cluster (the car) is label 0
    boxes = cpr.cluster_boxes(xyz, labels)
    car_box = boxes[0]
    assert abs(car_box['max'][0] - car_box['min'][0] - 4.5) < 0.1


def test_tiny_clusters_are_dropped():
    xyz = np.array([[0, 0, 1], [0.1, 0, 1], [5, 5, 1]], dtype=float)
    labels = cpr.cluster_2d(xyz[:, :2], tolerance=0.5, min_points=2)
    assert labels.tolist() == [0, 0, -1]


def test_laserscan_keeps_nearest_in_band():
    xyz = np.array([[5.0, 0.0, 0.3], [3.0, 0.0, 0.3], [2.0, 0.0, 3.0],   # 3.0 wins; 2.0 too high
                    [0.0, 4.0, 0.3]])                                     # 90 deg
    r = cpr.to_laserscan(xyz, -math.pi, math.pi, math.radians(1.0), 0.1, 50.0, 0.05, 2.0)
    assert math.isclose(r[180], 3.0)
    assert math.isclose(r[270], 4.0)
    assert np.isinf(r[0])
