"""Tests for the road / lane-line map (road_segmentation.py)."""

import numpy as np

from perceptron_perception import road_segmentation as rs

K = np.array([[400.0, 0, 319.5], [0, 400.0, 239.5], [0, 0, 1]])
# optical -> base: camera 0.42 m up looking along +x (optical z = base x, x = -y, y = -z)
M_BASE_OPT = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0.42], [0, 0, 0, 1.0]])


def test_labels_and_line_pixels_are_all_kept():
    ids = np.zeros((40, 40), np.uint8)
    ids[20:, :] = 1                      # road
    ids[30, 13] = 24                     # one marking pixel off the stride grid
    lab = rs.label_image(ids, rs.CARLA_IDS)
    v, u = rs.select_pixels(lab, stride=4)
    assert (30, 13) in set(zip(v.tolist(), u.tolist()))
    assert (lab[v, u] == rs.ROAD).sum() == 5 * 10


def test_depth_backprojection_and_ground_intersection_agree():
    u, v = np.array([319.5, 419.5]), np.array([339.5, 339.5])     # 100 px below the centre
    pts, ok = rs.ground_intersect(u, v, K, M_BASE_OPT)
    assert ok.all() and np.allclose(pts[:, 2], 0.0)
    assert np.isclose(pts[0, 0], 0.42 * 400 / 100) and np.isclose(pts[0, 1], 0.0)
    z = np.full(2, pts[0, 0])                                     # same planar depth
    opt = rs.backproject_depth(u, v, z, K)
    base = opt @ M_BASE_OPT[:3, :3].T + M_BASE_OPT[:3, 3]
    assert np.allclose(base, pts, atol=1e-9)


def test_rays_above_the_horizon_never_hit_the_ground():
    _, ok = rs.ground_intersect(np.array([319.5]), np.array([100.0]), K, M_BASE_OPT)
    assert not ok.any()


def test_map_keeps_line_over_road_and_rolls():
    m = rs.RoadMap(size=10.0, res=0.1, memory=5.0)
    m.recenter((0.0, 0.0))
    xyz = np.array([[1.03, 0.02, 0.0], [1.05, 0.05, 0.0], [2.0, 0.0, 0.1]])
    m.insert(xyz, np.array([rs.ROAD, rs.LINE, rs.ROAD], np.uint8), t=0.0)
    pts, lab = m.points()
    assert sorted(lab.tolist()) == [rs.ROAD, rs.LINE]            # one cell each, line wins its cell
    assert m.counts() == {'road': 1, 'sidewalk': 0, 'road_line': 1}
    m.recenter((3.0, 0.0))                                         # drove 3 m: cells stay in place
    pts, lab = m.points()
    assert np.allclose(sorted(pts[:, 0]), [1.05, 2.05])
    m.expire(10.0)
    assert not len(m.points()[0])


def test_packed_rgb_round_trip():
    f = rs.packed_rgb(np.array([rs.LINE], np.uint8))
    r, g, b = rs.LABEL_RGB[rs.LINE]
    assert f.view(np.uint32)[0] == (r << 16) | (g << 8) | b
