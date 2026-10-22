"""Road and lane-line map from a semantic segmentation image (no ROS).

Each labelled pixel becomes a 3D point, either from the depth image (CARLA) or, with no
depth, by intersecting its viewing ray with the ground plane z = 0 of base_footprint (the
real rover: a segmentation model on the camera + the flat-ground assumption, which holds
for markings, they ARE on the ground). Points go into a rolling grid in odom that keeps
the latest label per cell, so the road stays painted after it leaves the camera view.

Labels, in priority order when one cell gets several in the same frame: road_line >
sidewalk > road. Lane markings are a few pixels wide at distance, so their pixels are all
used; road and sidewalk pixels are subsampled.
"""

import numpy as np

from perceptron_perception.costmap import _shift

NONE, ROAD, SIDEWALK, LINE = 0, 1, 2, 3
LABEL_NAMES = {ROAD: 'road', SIDEWALK: 'sidewalk', LINE: 'road_line'}
# RViz colours (r, g, b): drivable road green, kerb magenta (Cityscapes sidewalk), marking orange
# (not white / yellow, so they stand apart from the CARLA map's lane lines drawn over them)
LABEL_RGB = {ROAD: (40, 170, 70), SIDEWALK: (200, 60, 200), LINE: (255, 140, 20)}

# CARLA 0.9.15 semantic tags; SIDEWALK stands for everything off the road
CARLA_IDS = {ROAD: (1,), SIDEWALK: (2, 9, 10, 25), LINE: (24,)}   # + vegetation, terrain, ground

# Nav2 cost per label, as OccupancyGrid values (Nav2 cost = value * 254 / 100):
# road 0, lane marking 30 (crossable, avoided), off-road 254 (lethal); unseen = 0 (free)
ROAD_COST_OCC = {ROAD: 0, LINE: 12, SIDEWALK: 100}


def label_image(ids, class_ids):
    """Semantic id image -> label image (NONE / ROAD / SIDEWALK / LINE)."""
    out = np.zeros(ids.shape, np.uint8)
    for label, values in class_ids.items():
        if values:
            out[np.isin(ids, np.asarray(values, ids.dtype))] = label
    return out


def select_pixels(labels, stride):
    """(v, u) of every LINE pixel and of every stride-th ROAD / SIDEWALK pixel."""
    v, u = np.nonzero(labels == LINE)
    sub = labels[::stride, ::stride]
    vs, us = np.nonzero((sub == ROAD) | (sub == SIDEWALK))
    return np.r_[v, vs * stride], np.r_[u, us * stride]


def backproject_depth(u, v, z, K):
    """Pixels + planar depth (CARLA's depth is along the optical axis) -> optical-frame xyz."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.column_stack([(u - cx) * z / fx, (v - cy) * z / fy, z])


def ground_intersect(u, v, K, M_base_opt):
    """Pixels -> where their rays meet z = 0 of the base frame; M_base_opt maps optical ->
    base. Returns (xyz in base, ok): rays at or above the horizon never reach the ground."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = np.column_stack([(u - cx) / fx, (v - cy) / fy, np.ones(len(u))]) @ M_base_opt[:3, :3].T
    o = M_base_opt[:3, 3]
    ok = d[:, 2] < -1e-6
    s = np.where(ok, -o[2] / np.where(ok, d[:, 2], -1.0), 0.0)
    return o + d * s[:, None], ok


class RoadMap:
    """Rolling grid (odom) of the latest road / sidewalk / road-line label per cell."""

    def __init__(self, size=40.0, res=0.1, memory=60.0):
        self.res = float(res)
        self.nx = self.ny = int(round(size / res))
        self.memory = float(memory)
        self.origin = None
        self.label = np.zeros((self.nx, self.ny), np.uint8)
        self.z = np.zeros((self.nx, self.ny), np.float32)
        self.seen = np.full((self.nx, self.ny), -np.inf)
        self.off_votes = np.zeros((self.nx, self.ny), np.uint8)   # frames in a row seen off-road

    def recenter(self, robot_xy):
        half = self.nx * self.res / 2.0
        want = np.floor((np.asarray(robot_xy, float) - half) / self.res) * self.res
        if self.origin is None:
            self.origin = want
            return
        shift = np.rint((want - self.origin) / self.res).astype(int)
        if np.all(np.abs(shift) < max(2, self.nx // 10)):
            return
        self.label = _shift(self.label, shift, NONE)
        self.z = _shift(self.z, shift, 0.0)
        self.seen = _shift(self.seen, shift, -np.inf)
        self.off_votes = _shift(self.off_votes, shift, 0)
        self.origin = self.origin + shift * self.res

    def insert(self, xyz, labels, t):
        """xyz (N, 3) in odom with their labels; the highest-priority label per cell wins
        within a frame, and replaces whatever the cell held before."""
        if self.origin is None or not len(xyz):
            return
        ij = np.floor((xyz[:, :2] - self.origin) / self.res).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)
        ij, lab, z = ij[ok], labels[ok], xyz[ok, 2]
        flat = ij[:, 0] * self.ny + ij[:, 1]
        best = np.zeros(self.nx * self.ny, np.uint8)
        np.maximum.at(best, flat, lab)
        cells = np.nonzero(best)[0]
        self.label.ravel()[cells] = best[cells]
        zsum = np.bincount(flat, weights=z, minlength=self.nx * self.ny)
        cnt = np.bincount(flat, minlength=self.nx * self.ny)
        self.z.ravel()[cells] = (zsum[cells] / cnt[cells]).astype(np.float32)
        self.seen.ravel()[cells] = t
        v = self.off_votes.ravel()
        off = best[cells] == SIDEWALK
        v[cells[off]] = np.minimum(v[cells[off]].astype(np.int16) + 1, 255).astype(np.uint8)
        v[cells[~off]] = 0

    def expire(self, t):
        old = (t - self.seen) > self.memory
        self.label[old] = NONE
        self.seen[old] = -np.inf
        self.off_votes[old] = 0

    def cost_grid(self, out_res=0.2, min_votes=3, min_blob=4):
        """OccupancyGrid values (int8, [ny_out, nx_out], rows along y) at out_res: the highest
        cost in each block. Off-road is lethal only after `min_votes` frames in a row and in
        blobs of >= `min_blob` cells, so one mislabelled frame or a speck cannot wall in the robot."""
        from scipy import ndimage
        occ = np.zeros((self.nx, self.ny), np.int8)
        occ[self.label == LINE] = ROAD_COST_OCC[LINE]
        occ[(self.label == SIDEWALK) & (self.off_votes >= min_votes)] = ROAD_COST_OCC[SIDEWALK]
        k = max(1, int(round(out_res / self.res)))
        n = self.nx // k
        occ = occ[:n * k, :n * k].reshape(n, k, n, k).max(axis=(1, 3))
        lethal = occ >= 100
        lab, nlab = ndimage.label(lethal)
        if nlab:
            sizes = ndimage.sum(lethal, lab, range(1, nlab + 1))
            small = np.isin(lab, np.flatnonzero(sizes < min_blob) + 1)
            occ[small] = 0
        return occ.T.copy()

    def points(self, labels=(ROAD, SIDEWALK, LINE)):
        """(xyz (M, 3) cell centres at their ground height, label (M,)) of labelled cells."""
        mask = np.isin(self.label, labels)
        i, j = np.nonzero(mask)
        xyz = np.column_stack([self.origin[0] + (i + 0.5) * self.res,
                               self.origin[1] + (j + 0.5) * self.res, self.z[i, j]])
        return xyz, self.label[i, j]

    def counts(self):
        return {name: int((self.label == k).sum()) for k, name in LABEL_NAMES.items()}


def packed_rgb(labels):
    """Labels -> float32 'rgb' field values (PCL / RViz packing: 0x00RRGGBB as float bits)."""
    lut = np.zeros(256, np.uint32)
    for k, (r, g, b) in LABEL_RGB.items():
        lut[k] = (r << 16) | (g << 8) | b
    return lut[labels].view(np.float32)
