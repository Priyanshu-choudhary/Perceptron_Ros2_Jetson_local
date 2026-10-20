"""Camera boxes + lidar clusters -> objects with class, 3D position and range (no ROS).

Stage 5 of the roadmap. Testable on its own -- see test/test_object_fusion.py.

    obstacle points (base_footprint, with Stage 4 cluster labels)
      --project (camera_projection, the same chain as the overlay)--> pixels
      --associate with each 2D box--> point set per box
      --> centroid, extent, nearest-surface range, covariance

ASSOCIATION: CLUSTERS FIRST, HORIZONTAL OVERLAP

Taking "every point inside the box" fails twice over: a box is a frustum, so it
also holds the road and the wall behind the object, and a pedestrian standing
in front of a car is inside the car's box. Stage 4 already split the scene
into objects, so the unit of association is a cluster. Each cluster is scored
against each box by

    affinity = horizontal IoU(cluster's u-span, box's u-span)
               x (share of the cluster's projected points inside the box)

Horizontal only: the rover lidar sees the bottom 0.42 m of things, so a
cluster is a thin strip along the bottom of its box, and a full 2D IoU would
always be small. A pedestrian's strip spans the pedestrian's box but only a
sliver of a car's box, so it goes to the pedestrian. Each cluster goes to its
best box (affinity >= min_affinity); a box may take several clusters. A
second pass re-joins what Stage 4 split off an object -- a car's narrow side
face fails the overlap test but lies entirely in the car's box -- when it is
inside exactly one box and within join_depth of that box's main cluster.

A box that gets no cluster (an object too sparse to cluster at range) falls
back to the loose points inside it: the nearest depth group of at least
fallback_min_points, split where consecutive depths jump by more than
depth_gap.

RANGE IS TO THE NEAREST SURFACE

The lidar only sees the near side of an object, so the centroid of its points
is biased towards the sensor by up to half the object's depth. What the robot
needs for braking is the distance to the closest surface, so `range` is the
5th percentile of point distances from the camera (robust to one stray point)
and is what the evaluator compares with the true closest point of the object's
3D box. The centroid is still published as the object's position.
"""

import numpy as np

from perceptron_navigation import camera_projection as cp


def project(pts, M_optical_from_points, K, D, width, height, min_z=0.15, max_radius=2.35):
    """Pixels of points that land in the image: (uv (M,2), depth (M,), index (M,))."""
    opt = cp.transform_points(M_optical_from_points, pts)
    vis = cp.visible_mask(opt, min_z, max_radius)
    idx = np.flatnonzero(vis)
    uv = cp.project_points(opt[idx], K, D)
    inside = cp.inside_image(uv, width, height)
    return uv[inside], opt[idx[inside], 2], idx[inside]


def shrink_box(box, shrink):
    """(x0, y0, x1, y1) narrowed by `shrink` of its width on each side (height kept)."""
    x0, y0, x1, y1 = box
    dx = (x1 - x0) * shrink
    return x0 + dx, y0, x1 - dx, y1


def _hiou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def associate(boxes, uv, depth, idx, labels, min_affinity=0.2, shrink=0.05,
              fallback_min_points=2, depth_gap=1.0, min_span_px=4.0, join_share=0.8, join_depth=2.5):
    """Point indices (into the full cloud) for each box; empty array when nothing fits.

    boxes   list of (x0, y0, x1, y1) in pixels
    uv, depth, idx   from project(); labels: cluster label per FULL-cloud point, -1 = none
    """
    nb = len(boxes)
    out = [np.zeros(0, dtype=np.int64) for _ in range(nb)]
    if nb == 0 or len(idx) == 0:
        return out, [None] * nb
    sboxes = [shrink_box(b, shrink) for b in boxes]
    in_box = np.zeros((nb, len(idx)), dtype=bool)
    for i, (x0, y0, x1, y1) in enumerate(sboxes):
        in_box[i] = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)

    lab = labels[idx]
    how = [None] * nb
    main_depth = [[] for _ in range(nb)]
    leftovers = []
    for c in np.unique(lab[lab >= 0]):
        sel = lab == c
        u = uv[sel, 0]
        c0, c1 = float(u.min()), float(u.max())
        if c1 - c0 < min_span_px:            # a far, thin object can project to one column
            mid = (c0 + c1) / 2.0
            c0, c1 = mid - min_span_px / 2.0, mid + min_span_px / 2.0
        best, best_aff = -1, min_affinity
        for i, (x0, _, x1, _) in enumerate(sboxes):
            n_in = int((in_box[i] & sel).sum())
            if n_in == 0:
                continue
            aff = _hiou(c0, c1, x0, x1) * n_in / sel.sum()
            if aff > best_aff:
                best, best_aff = i, aff
        if best >= 0:
            out[best] = np.concatenate([out[best], np.flatnonzero(labels == c)])
            how[best] = 'cluster'
            main_depth[best].append(float(np.median(depth[sel])))
        else:
            leftovers.append(c)

    # Second pass: a narrow cluster (a car's side face, a wheel) has a small horizontal
    # overlap with its object's box and fails the first test, yet sits entirely inside it.
    # Join it to the one box that contains it if its depth is within join_depth of that
    # box's main cluster -- which a pedestrian standing well in front of a car is not.
    for c in leftovers:
        sel = lab == c
        holders = [i for i in range(nb) if (in_box[i] & sel).sum() >= join_share * sel.sum()]
        if len(holders) != 1 or not main_depth[holders[0]]:
            continue
        i = holders[0]
        if abs(float(np.median(depth[sel])) - min(main_depth[i])) <= join_depth:
            out[i] = np.concatenate([out[i], np.flatnonzero(labels == c)])

    for i in range(nb):
        if len(out[i]) or how[i] is not None:
            continue
        sel = in_box[i] & (lab < 0)
        if sel.sum() < fallback_min_points:
            continue
        d = depth[sel]
        order = np.argsort(d)
        ds = d[order]
        cut = np.flatnonzero(np.diff(ds) > depth_gap)
        end = cut[0] + 1 if len(cut) else len(ds)
        if end >= fallback_min_points:
            out[i] = idx[np.flatnonzero(sel)[order[:end]]]
            how[i] = 'loose points'
    return out, how


def describe(pts, sensor_origin, noise_floor=0.05):
    """Geometry of one object's points (base_footprint): dict, or None without points."""
    if len(pts) == 0:
        return None
    dist = np.linalg.norm(pts - sensor_origin, axis=1)
    rng = float(np.percentile(dist, 5))
    centroid = pts.mean(axis=0)
    cov = np.cov(pts.T) / len(pts) if len(pts) > 1 else np.zeros((3, 3))
    cov = cov + np.eye(3) * noise_floor ** 2
    return {'n': int(len(pts)), 'centroid': centroid, 'min': pts.min(axis=0), 'max': pts.max(axis=0),
            'range': rng, 'cov': cov}
