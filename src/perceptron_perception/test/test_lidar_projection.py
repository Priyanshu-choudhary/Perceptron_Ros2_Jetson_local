"""Tests for the lidar overlay geometry and the depth-consistency pixel error.

test_alignment_recovers_known_shifts_including_sideways is the one that
matters: it renders a floor and some boxes into a depth image, offsets the
projected points by a known vector, and checks the accumulator reports that
vector. If it did not, every number the projection check prints would be
meaningless.
"""

import math
from types import SimpleNamespace

import numpy as np

from perceptron_navigation import camera_projection as cp
from perceptron_perception import lidar_projection as lp

W, H = 640, 480
F = 320.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
NO_DIST = np.zeros(5)
CAM_HEIGHT = 1.5


def ground_depth_image():
    """Planar depth of a flat floor 1.5 m below a level camera; sky = 1000 m."""
    v = np.arange(H, dtype=np.float64)[:, None] - K[1, 2]
    with np.errstate(divide='ignore'):
        z = np.where(v > 0, F * CAM_HEIGHT / np.maximum(v, 1e-9), 1000.0)
    return np.broadcast_to(z, (H, W)).astype(np.float32)


def cloud_msg(xyz, extra_field=True):
    """A PointCloud2-shaped object with x,y,z,intensity float32 fields."""
    fields = [SimpleNamespace(name=n, offset=4 * i, datatype=7, count=1)
              for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
    data = np.column_stack([xyz, np.zeros(len(xyz))]).astype(np.float32)
    return SimpleNamespace(fields=fields, width=len(xyz), height=1, point_step=16,
                           is_bigendian=False, data=data.tobytes())


def test_cloud_to_xyz_reads_fields_and_drops_nan():
    xyz = np.array([[1.0, 2.0, 3.0], [np.nan, 0, 0], [4.0, 5.0, 6.0]])
    out = lp.cloud_to_xyz(cloud_msg(xyz))
    np.testing.assert_allclose(out, [[1, 2, 3], [4, 5, 6]])


def test_image_to_bgr_handles_rgb_and_padding():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[..., 0] = 200                                   # red in RGB
    padded = np.zeros((2, 12), dtype=np.uint8)          # step 12 > width*3 = 9
    padded[:, :9] = rgb.reshape(2, 9)
    msg = SimpleNamespace(encoding='rgb8', height=2, width=3, step=12, data=padded.tobytes())
    bgr = lp.image_to_bgr(msg)
    assert bgr.shape == (2, 3, 3)
    assert (bgr[..., 2] == 200).all() and (bgr[..., 0] == 0).all()


def test_depth_to_array_32fc1():
    d = np.arange(6, dtype=np.float32).reshape(2, 3)
    msg = SimpleNamespace(encoding='32FC1', height=2, width=3, step=12, data=d.tobytes())
    np.testing.assert_array_equal(lp.depth_to_array(msg), d)


def test_intrinsics_from_camera_info_pads_distortion():
    msg = SimpleNamespace(k=list(K.ravel()), d=[], width=W, height=H)
    K2, D2, size = lp.intrinsics_from_camera_info(msg)
    np.testing.assert_array_equal(K2, K)
    assert D2.shape == (5,) and not D2.any() and size == (W, H)


def scene_depth_image():
    """Floor plus three upright boxes at 6, 12 and 25 m: vertical edges pin du, the floor pins dv."""
    depth = ground_depth_image().copy()
    for (u0, u1, v0, v1, z) in ((80, 160, 150, 300, 6.0), (300, 340, 200, 260, 12.0),
                                (480, 500, 220, 250, 25.0)):
        depth[v0:v1, u0:u1] = z
    return depth


def scene_points(depth, n=6000, seed=1):
    """Sample visible scene pixels and use the rendered depth as the point's Z (no sky)."""
    rng = np.random.default_rng(seed)
    v = rng.integers(K[1, 2] + 2, H - 2, n)
    u = rng.integers(2, W - 2, n)
    keep = depth[v, u] < 100.0
    u, v = u[keep], v[keep]
    return np.column_stack([u, v]).astype(np.float64), depth[v, u].astype(np.float64)


def test_alignment_reads_zero_when_aligned():
    depth = scene_depth_image()
    uv, z = scene_points(depth)
    acc = lp.AlignmentAccumulator(bins=(0, 1000), max_shift=8)
    acc.add(uv, z, z, depth)
    r = acc.results()[0]
    assert r['err_px'] < 0.5, r
    assert r['match_zero_pct'] > 95.0


def test_alignment_recovers_known_shifts_including_sideways():
    # (11.2, 0) is what a 1 deg camera yaw does at f = 640. A sideways error is
    # exactly what a nearest-consistent-pixel metric cannot see on the floor.
    depth = scene_depth_image()
    uv, z = scene_points(depth)
    for e in ((3.0, 0.0), (-2.0, 2.0), (0.0, -4.0), (11.2, 0.0)):
        acc = lp.AlignmentAccumulator(bins=(0, 1000), max_shift=15)
        acc.add(uv + e, z, z, depth)
        r = acc.results()[0]
        assert abs(r['du'] - e[0]) <= 0.6 and abs(r['dv'] - e[1]) <= 0.6, (e, r)
        assert not r['at_edge']


def test_alignment_flags_error_bigger_than_search():
    depth = scene_depth_image()
    uv, z = scene_points(depth)
    acc = lp.AlignmentAccumulator(bins=(0, 1000), max_shift=4)
    acc.add(uv + (9.0, 0.0), z, z, depth)
    assert acc.results()[0]['at_edge']


def test_alignment_merge_and_bins():
    depth = scene_depth_image()
    uv, z = scene_points(depth)
    a = lp.AlignmentAccumulator(bins=(0, 10, 1000), max_shift=3)
    b = lp.AlignmentAccumulator(bins=(0, 10, 1000), max_shift=3)
    a.add(uv, z, z, depth)
    b.add(uv, z, z, depth)
    a.merge(b)
    assert a.frames == 2
    assert a.points.sum() == 2 * len(z)
    assert a.points[0] == 2 * int((z < 10).sum())
    assert 'range' in lp.format_alignment(a.results(), 't')


def test_perturbation_yaw_shifts_by_f_tan():
    pt = np.array([[0.0, 0.0, 10.0]])
    M = lp.perturbation([0.0, 1.0, 0.0], [0, 0, 0])
    du = (cp.project_points(cp.transform_points(M, pt), K, NO_DIST)
          - cp.project_points(pt, K, NO_DIST))[0, 0]
    assert abs(abs(du) - F * math.tan(math.radians(1.0))) < 0.05


def test_draw_points_puts_near_on_top():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    uv = np.array([[5.0, 5.0], [5.0, 5.0]])
    colours = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)   # far blue, near red
    lp.draw_points(img, uv, colours, np.array([20.0, 2.0]), radius=0)
    assert img[5, 5].tolist() == [0, 0, 255]
