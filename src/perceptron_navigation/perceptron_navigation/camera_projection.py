"""Points in any TF frame -> camera pixels, shared by every camera overlay.

    pts (frame F) --T_optical_F--> camera_optical_link --cull--> projectPoints -> (u, v)

Deliberately free of ROS imports, like marker_map.py, so the geometry can be
tested without a running graph -- see test/test_camera_projection.py. Used by
path_overlay_node (the Nav2 plan) and perceptron_perception's
lidar_camera_overlay_node (3D lidar points); both get the same intrinsics, the
same URDF chain and the same cull, so a fix to one is a fix to both.

WHY THE CULL IS TWO-STAGE AND NOT JUST Z > 0

The lens is ~115 deg horizontal (fx ~= 411 over 1280 px), and a 5-coefficient
plumb-bob model is only valid inside the cone it was fitted in. Differentiating
the radial polynomial with this calibration's coefficients:

    r = 1.76 (image corner)  ->  d/dr [r*(1 + k1 r^2 + k2 r^4 + k3 r^6)] = +1.07
    r = 3.0  (outside FOV)   ->  the same derivative = -1.29

Past r ~= 2.4 the model folds: points well outside the field of view come back
with plausible pixel coordinates INSIDE the image. A path sweeping past the
edge of frame while the robot turns does exactly that, and the symptom is a
phantom stripe that whips across the picture. A lidar cloud is worse: it has
points all the way round the robot, and every one of them past the fold lands
somewhere in the frame. Z > 0 does not catch it because those points are
genuinely in front of the camera, just not in view. So the mask is
Z > min_z AND r <= max_radius, applied before projectPoints.
"""

import math

import cv2
import numpy as np


def quaternion_to_rotation(x, y, z, w):
    """3x3 rotation from a quaternion. Normalised first; TF is not exact."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError('zero-length quaternion')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_to_matrix(transform_stamped):
    """geometry_msgs/TransformStamped (or anything shaped like it) -> 4x4."""
    q = transform_stamped.transform.rotation
    t = transform_stamped.transform.translation
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = quaternion_to_rotation(q.x, q.y, q.z, q.w)
    M[:3, 3] = (t.x, t.y, t.z)
    return M


def transform_points(M, pts):
    """Apply a 4x4 to (N, 3) points."""
    return pts @ M[:3, :3].T + M[:3, 3]


def visible_mask(pts_opt, min_z, max_radius, max_range=None):
    """Two-stage cull in the optical frame. See the module docstring for why r matters."""
    Z = pts_opt[:, 2]
    ok = Z > min_z
    if not ok.any():
        return ok
    Zs = np.where(ok, Z, 1.0)
    r = np.hypot(pts_opt[:, 0] / Zs, pts_opt[:, 1] / Zs)
    ok &= np.isfinite(r) & (r <= max_radius)
    if max_range is not None:
        ok &= np.linalg.norm(pts_opt, axis=1) <= max_range
    return ok


def project_points(pts_opt, camera_matrix, dist_coeffs):
    """Optical-frame points -> (N, 2) float pixel coords. Cull first."""
    if len(pts_opt) == 0:
        return np.empty((0, 2), dtype=np.float64)
    uv, _ = cv2.projectPoints(
        np.ascontiguousarray(pts_opt, dtype=np.float64).reshape(-1, 1, 3),
        np.zeros(3), np.zeros(3),
        camera_matrix, dist_coeffs)
    return uv.reshape(-1, 2)


def project_pixels(pts_opt, camera_matrix, dist_coeffs):
    """Same as project_points, rounded to int32 for drawing."""
    return np.rint(project_points(pts_opt, camera_matrix, dist_coeffs)).astype(np.int32)


def inside_image(uv, width, height):
    """Mask of pixel coords that land on a width x height image."""
    return ((uv[:, 0] >= 0) & (uv[:, 0] < width) &
            (uv[:, 1] >= 0) & (uv[:, 1] < height))
