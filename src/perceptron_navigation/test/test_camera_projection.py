"""Geometry tests for the shared camera projection.

test_point_ahead_lands_on_principal_point and test_lidar_chain_round_trip are
the important ones: they push points through the same URDF chain the robot
uses (lidar3d_link -> base_link -> camera_link -> camera_optical_link) and
check they land where a hand-written pinhole says they should, which is what
catches a swapped axis in the optical frame.
"""

import math
from types import SimpleNamespace

import numpy as np

from perceptron_navigation import camera_projection as cp
from perceptron_navigation import marker_map as mm

CAMERA_MATRIX = np.array([
    [410.9212916385953, 0.0, 634.8042096493966],
    [0.0, 411.4395116912116, 347.58359871233904],
    [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_COEFFS = np.array([0.030287865182229395, -0.002860707267771121,
                        -0.004223044189411946, -0.0003374289867825264,
                        -0.0003820853552867496], dtype=np.float64)
NO_DIST = np.zeros(5)


def rpy_matrix(roll, pitch, yaw):
    """URDF rpy: R = Rz(yaw) . Ry(pitch) . Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp_, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp_, 0, sp], [0, 1, 0], [-sp, 0, cp_]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def base_from_optical():
    """base_link <- camera_optical_link as perceptron_robot.xacro builds it."""
    base_from_camera = mm.make_transform(rpy_matrix(0, -0.07027, 0), (0, 0, 0.42 - 0.0625))
    camera_from_optical = mm.make_transform(
        rpy_matrix(-math.pi / 2, 0, -math.pi / 2), (0, 0, 0))
    return base_from_camera @ camera_from_optical


def base_from_lidar3d():
    """base_link <- lidar3d_link: on the lidar axis, 0.075 m up, pitched 0.26 rad."""
    return mm.make_transform(rpy_matrix(0, 0.26, 0), (0, 0, 0.2775 + 0.075))


def test_quaternion_matches_optical_joint():
    # rpy(-pi/2, 0, -pi/2) as a quaternion is (-0.5, 0.5, -0.5, 0.5).
    R = cp.quaternion_to_rotation(-0.5, 0.5, -0.5, 0.5)
    np.testing.assert_allclose(R, rpy_matrix(-math.pi / 2, 0, -math.pi / 2), atol=1e-12)
    # Optical +Z (out of the lens) is camera +X (forward).
    np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-12)


def test_transform_to_matrix_reads_a_transform_stamped():
    msg = SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=2.0)))   # unnormalised on purpose
    np.testing.assert_allclose(cp.transform_to_matrix(msg),
                               mm.make_transform(np.eye(3), (1, 2, 3)))


def test_point_ahead_lands_on_principal_point():
    # A point straight down the optical axis, 3 m out.
    uv = cp.project_points(np.array([[0.0, 0.0, 3.0]]), CAMERA_MATRIX, DIST_COEFFS)
    np.testing.assert_allclose(uv[0], CAMERA_MATRIX[:2, 2], atol=1e-9)


def test_lidar_chain_round_trip():
    # Points in front of the robot at 1..20 m, a little left and above the floor.
    optical_from_lidar = mm.invert_transform(base_from_optical()) @ base_from_lidar3d()
    rng = np.random.default_rng(0)
    pts_base = np.column_stack([rng.uniform(1, 20, 200), rng.uniform(-2, 2, 200),
                                rng.uniform(0.0, 1.5, 200)])
    pts_lidar = cp.transform_points(mm.invert_transform(base_from_lidar3d()), pts_base)

    pts_opt = cp.transform_points(optical_from_lidar, pts_lidar)
    uv = cp.project_points(pts_opt, CAMERA_MATRIX, NO_DIST)

    # Hand-written pinhole straight from base_link, no shared helpers.
    opt = cp.transform_points(mm.invert_transform(base_from_optical()), pts_base)
    u = CAMERA_MATRIX[0, 0] * opt[:, 0] / opt[:, 2] + CAMERA_MATRIX[0, 2]
    v = CAMERA_MATRIX[1, 1] * opt[:, 1] / opt[:, 2] + CAMERA_MATRIX[1, 2]
    np.testing.assert_allclose(uv, np.column_stack([u, v]), atol=1e-6)

    # Left of the robot is left in the image; above the lens is up in the image.
    left = np.array([[5.0, 1.0, 0.3575]])
    above = np.array([[5.0, 0.0, 1.5]])
    T = mm.invert_transform(base_from_optical())
    centre = cp.project_points(cp.transform_points(T, np.array([[5.0, 0, 0.3575]])),
                               CAMERA_MATRIX, NO_DIST)[0]
    assert cp.project_points(cp.transform_points(T, left), CAMERA_MATRIX, NO_DIST)[0, 0] < centre[0]
    assert cp.project_points(cp.transform_points(T, above), CAMERA_MATRIX, NO_DIST)[0, 1] < centre[1]


def test_cull_drops_behind_and_fold_back():
    pts = np.array([
        [0.0, 0.0, 2.0],    # dead ahead            -> keep
        [0.0, 0.0, -2.0],   # behind the camera     -> drop
        [6.0, 0.0, 2.0],    # r = 3, past the fold  -> drop
        [0.0, 0.0, 0.05],   # closer than min_z     -> drop
        [0.0, 0.0, 50.0],   # beyond max_range      -> drop
    ])
    mask = cp.visible_mask(pts, min_z=0.15, max_radius=2.35, max_range=8.0)
    assert mask.tolist() == [True, False, False, False, False]
    # With no range limit the far point survives.
    assert cp.visible_mask(pts, 0.15, 2.35)[4]


def test_fold_back_is_real_with_this_calibration():
    # Why the radius cull exists: r = 3.5 is ~74 deg off-axis, well outside the
    # ~57 deg half-FOV, yet the folded model maps it to r_d ~= 0.84, inside the image.
    pt = np.array([[3.5, 0.0, 1.0]])
    uv = cp.project_points(pt, CAMERA_MATRIX, DIST_COEFFS)
    assert cp.inside_image(uv, 1280, 720)[0]
    assert not cp.visible_mask(pt, 0.15, 2.35)[0]


def test_inside_image_edges():
    uv = np.array([[0, 0], [1279, 719], [1280, 10], [-1, 10]], dtype=np.float64)
    assert cp.inside_image(uv, 1280, 720).tolist() == [True, True, False, False]
