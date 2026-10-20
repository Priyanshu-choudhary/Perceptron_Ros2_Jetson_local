"""3D lidar preprocessing: filter, remove the ground, cluster, flatten to a scan.

No ROS imports, so every step is testable on synthetic clouds -- see
test/test_cloud_processing.py. lidar_preprocessor_node wires these together:

    cloud (lidar frame) --range--> base_footprint --self box--> voxel
        --> ground / obstacles --> clusters (+ boxes) and a LaserScan

Everything after the range filter works in base_footprint: z up, origin on the
ground under the robot. The lidar is pitched 15 deg down, so its own frame has
no useful "up"; base_footprint does, which is what makes a height threshold or
a 2D cluster mean something.

GROUND: WHY A PLANE PLUS A HEIGHT GRID

A single RANSAC plane finds the road, but a raised sidewalk sits 10-20 cm above
it and would come out as one enormous obstacle. A pure local-minimum grid
handles steps, but it calls the bottom of a car ground wherever the lidar never
saw the road under it. So a point is ground if EITHER it is within
plane_threshold of the fitted plane, OR it is within cell_height of the lowest
point of its grid cell AND that lowest point is itself within cell_plane_tol of
the plane (so a cell whose lowest point is a car's roof does not count). The
plane is also the extrinsic check of roadmap task 4.1: with a correct lidar
mount it comes out at z ~ 0 and tilt ~ 0 in base_footprint.

CLUSTERS ARE 2D

Obstacle points are clustered on x, y only. A 16-beam lidar's rings are 2 deg
apart, 0.7 m vertically at 20 m, so a 3D distance threshold small enough to
keep two people apart also cuts one person into ring-shaped slices. Seen from
above, an object's rings collapse onto one footprint.
"""

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- filters

def range_mask(xyz_sensor, min_range, max_range):
    """Points whose distance from the sensor is inside [min_range, max_range]."""
    r = np.linalg.norm(xyz_sensor, axis=1)
    return (r >= min_range) & (r <= max_range)


def box_mask(xyz, box):
    """Points inside an axis-aligned box [x0, x1, y0, y1, z0, z1]."""
    x0, x1, y0, y1, z0, z1 = box
    return ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1) &
            (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1))


def voxel_downsample(xyz, voxel):
    """Indices of one point per occupied voxel (the first seen). voxel <= 0 keeps all."""
    if voxel <= 0 or len(xyz) == 0:
        return np.arange(len(xyz))
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)


# ---------------------------------------------------------------- ground

def fit_plane_ransac(xyz, threshold=0.1, iterations=60, max_tilt_deg=20.0, rng=None):
    """Best plane (n, d) with n.p + d = 0, n.z > 0, tilt <= max_tilt_deg; or None."""
    if len(xyz) < 3:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    cos_max = math.cos(math.radians(max_tilt_deg))
    best, best_count = None, 0
    for _ in range(iterations):
        a, b, c = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if n[2] < 0:
            n = -n
        if n[2] < cos_max:
            continue
        d = -float(n @ a)
        count = int(np.count_nonzero(np.abs(xyz @ n + d) < threshold))
        if count > best_count:
            best, best_count = (n, d), count
    if best is None:
        return None
    # least-squares refit on the inliers: RANSAC's three points are noisy
    n, d = best
    inl = xyz[np.abs(xyz @ n + d) < threshold]
    if len(inl) >= 3:
        centroid = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
        n = vt[2] if vt[2][2] > 0 else -vt[2]
        d = -float(n @ centroid)
    return n, d


def plane_tilt_deg(n):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))


def plane_height_at_origin(plane):
    """z where the plane crosses the base_footprint z axis (0 for a correct mount)."""
    n, d = plane
    return -d / n[2]


def segment_ground(xyz, plane_band=0.6, plane_threshold=0.15, cell=0.5, cell_height=0.12,
                   cell_plane_tol=0.3, ransac_threshold=0.05, rng=None):
    """Ground mask and fitted plane for points in a z-up frame (base_footprint).

    plane_band       only points with |z| < plane_band are offered to RANSAC
    ransac_threshold inlier distance for FITTING the plane. Kept tight on purpose: at 10 cm
                     RANSAC happily bridges a road and a 15 cm sidewalk with a plane tilted
                     a degree, which would also fool the extrinsic check
    plane_threshold  a point this close to the fitted plane is ground
    cell, cell_height, cell_plane_tol: the local-minimum grid (see module docstring)
    """
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool), None
    cand = xyz[np.abs(xyz[:, 2]) < plane_band]
    plane = fit_plane_ransac(cand, threshold=ransac_threshold, rng=rng)
    if plane is None:
        # no plane: fall back to "near z = 0", which is what base_footprint means
        plane = (np.array([0.0, 0.0, 1.0]), 0.0)
    n, d = plane
    height = xyz @ n + d                                    # signed height above the plane
    ground = height < plane_threshold

    keys = np.floor(xyz[:, :2] / cell).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    cell_min = np.full(inv.max() + 1, np.inf)
    np.minimum.at(cell_min, inv, height)
    local = height - cell_min[inv]
    ground |= (local < cell_height) & (np.abs(cell_min[inv]) < cell_plane_tol)
    return ground, plane


# ---------------------------------------------------------------- clusters

def cluster_2d(xy, tolerance, min_points):
    """Euclidean clustering on x, y. Labels 0..k-1 by descending size, -1 = too small."""
    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pairs = cKDTree(xy).query_pairs(tolerance, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    sizes = np.bincount(comp)
    order = np.argsort(-sizes, kind='stable')
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    labels = rank[comp]
    labels[sizes[comp] < min_points] = -1
    return labels.astype(np.int64)


def cluster_boxes(xyz, labels):
    """One dict per cluster: id, n, centroid, min, max (axis-aligned in the input frame)."""
    out = []
    if len(labels) == 0:
        return out
    for k in range(labels.max() + 1):
        p = xyz[labels == k]
        if len(p) == 0:
            continue
        out.append({'id': k, 'n': len(p), 'centroid': p.mean(axis=0),
                    'min': p.min(axis=0), 'max': p.max(axis=0)})
    return out


# ---------------------------------------------------------------- scan

def to_laserscan(xyz, angle_min, angle_max, angle_increment, range_min, range_max,
                 z_min, z_max):
    """Nearest obstacle per bearing for points with z in [z_min, z_max]; inf = nothing."""
    nbins = int(round((angle_max - angle_min) / angle_increment)) + 1
    ranges = np.full(nbins, np.inf, dtype=np.float32)
    band = (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    p = xyz[band]
    if len(p) == 0:
        return ranges
    r = np.hypot(p[:, 0], p[:, 1])
    a = np.arctan2(p[:, 1], p[:, 0])
    ok = (r >= range_min) & (r <= range_max) & (a >= angle_min) & (a <= angle_max)
    b = np.rint((a[ok] - angle_min) / angle_increment).astype(np.int64)
    np.minimum.at(ranges, np.clip(b, 0, nbins - 1), r[ok].astype(np.float32))
    return ranges
