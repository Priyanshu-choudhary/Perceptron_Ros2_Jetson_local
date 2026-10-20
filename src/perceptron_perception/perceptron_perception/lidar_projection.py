"""Lidar-on-camera geometry: decode, colour, draw, and measure the alignment.

No ROS imports (messages are read duck-typed), so all of it is testable
without a graph -- see test/test_lidar_projection.py. The projection itself is
perceptron_navigation.camera_projection, the same code the path overlay uses.

HOW THE ALIGNMENT IS MEASURED (AlignmentAccumulator)

A projected lidar point at pixel (u, v) with optical depth Z is correct if the
camera sees something at Z there. In CARLA a depth camera at the RGB camera's
exact pose renders that Z for every pixel. So for every candidate image shift
(du, dv) within +/- max_shift we count how many points' depths agree with the
depth image sampled at (u + du, v + dv), per range bin, summed over frames.
The shift with the most agreement is where the points SHOULD have landed; the
alignment error is minus that shift, refined to sub-pixel with a parabola.
0 when the extrinsic and intrinsics are right.

WHY ONE SHIFT PER BIN AND NOT A PER-POINT ERROR

The obvious per-point metric -- distance to the nearest pixel with matching
depth -- is blind to any error along a surface. A 1 deg yaw moves every point
~11 px sideways, but along a row of the ground the depth barely changes, so
the point still finds its depth at offset 0 and scores a perfect 0. It only
ever sees the component of the error across the local depth gradient. A
global shift is different: vertical edges (poles, cones, walls) pin du, the
ground pins dv, and points that cannot tell (flat surfaces, occlusions where
the two sensors see different things, CARLA's coarse lidar collision meshes)
add the same count at every shift and so cannot move the peak.
"""

import cv2
import numpy as np

_FLOAT32 = 7   # sensor_msgs/PointField.FLOAT32

DEFAULT_RANGE_BINS = (0.0, 5.0, 10.0, 20.0, 40.0, 100.0)


# --------------------------------------------------------------- decoding

def cloud_to_xyz(msg):
    """sensor_msgs/PointCloud2 -> (N, 3) float64, non-finite points dropped."""
    fields = {f.name: f for f in msg.fields}
    for name in ('x', 'y', 'z'):
        if name not in fields:
            raise ValueError('cloud has no "%s" field' % name)
        if fields[name].datatype != _FLOAT32:
            raise ValueError('cloud field "%s" is not FLOAT32' % name)
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    dtype = np.dtype({'names': ['x', 'y', 'z'],
                      'formats': ['<f4' if not msg.is_bigendian else '>f4'] * 3,
                      'offsets': [fields[c].offset for c in 'xyz'],
                      'itemsize': msg.point_step})
    raw = np.frombuffer(msg.data, dtype=dtype, count=n)
    xyz = np.column_stack([raw['x'], raw['y'], raw['z']]).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def cloud_fields(msg, names):
    """Named float32 columns of a PointCloud2, as a structured array (no copy)."""
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({'names': names, 'formats': ['<f4'] * len(names),
                   'offsets': [off[n] for n in names], 'itemsize': msg.point_step})
    return np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)


def image_to_bgr(msg):
    """sensor_msgs/Image (bgr8, rgb8, bgra8, rgba8, mono8) -> HxWx3 uint8 BGR copy."""
    enc = msg.encoding.lower()
    channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, 'mono8': 1}.get(enc)
    if channels is None:
        raise ValueError('unsupported image encoding %s' % msg.encoding)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = buf.reshape(msg.height, msg.step)[:, :msg.width * channels]
    img = img.reshape(msg.height, msg.width, channels)
    if enc == 'bgr8':
        return img.copy()
    if enc == 'rgb8':
        return img[:, :, ::-1].copy()
    if enc == 'bgra8':
        return img[:, :, :3].copy()
    if enc == 'rgba8':
        return img[:, :, 2::-1].copy()
    return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)


def intrinsics_from_camera_info(msg):
    """sensor_msgs/CameraInfo -> (K 3x3, dist (5,), (width, height))."""
    K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
    D = np.zeros(5, dtype=np.float64)
    d = np.asarray(msg.d, dtype=np.float64)
    D[:min(5, d.size)] = d[:5]
    return K, D, (int(msg.width), int(msg.height))


def perturbation(rpy_deg, xyz):
    """4x4 applied in the optical frame, to fake a mis-calibrated extrinsic.

    Exists so the accuracy check can be shown to catch an error: rpy_deg
    (0, 1, 0) -- about optical +Y, a camera yaw -- at fx ~= 411 must come out
    as ~7 px, or the check is not measuring anything.
    """
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=np.float64))
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = rz @ ry @ rx
    M[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return M


def depth_to_array(msg):
    """sensor_msgs/Image 32FC1 (metres) or 16UC1 (millimetres) -> HxW float32 metres."""
    enc = msg.encoding.upper()
    if enc == '32FC1':
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)
        return d[:, :msg.width]
    if enc == '16UC1':
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
        return d[:, :msg.width].astype(np.float32) / 1000.0
    raise ValueError('unsupported depth encoding %s' % msg.encoding)


# ---------------------------------------------------------------- drawing

def depth_colours(z, max_depth):
    """(N,) depths -> (N, 3) uint8 BGR, near = red, far = blue (JET reversed)."""
    scaled = np.clip(255.0 * (1.0 - np.clip(z, 0.0, max_depth) / max_depth), 0, 255)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
    return lut[scaled.astype(np.uint8), 0, :]


def draw_points(img, uv, colours, z, radius):
    """Splat square dots into img in place, far first so near points stay on top."""
    h, w = img.shape[:2]
    order = np.argsort(-z)
    u = np.rint(uv[order, 0]).astype(np.int64)
    v = np.rint(uv[order, 1]).astype(np.int64)
    c = colours[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[ok], uu[ok]] = c[ok]
    return img


# ------------------------------------------------------------- measuring

def sample_bilinear(img, u, v):
    """Bilinear sample of a 2-D float image at pixel-centre coords; (values, in_bounds).

    Rounding to the nearest pixel is not good enough here: on the ground far
    ahead the rendered depth changes by metres per pixel, so a perfectly
    aligned point would miss its own depth by more than any sane tolerance.
    """
    h, w = img.shape
    inb = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    uc = np.clip(u, 0, w - 1)
    vc = np.clip(v, 0, h - 1)
    u0 = np.minimum(np.floor(uc).astype(np.int64), w - 2)
    v0 = np.minimum(np.floor(vc).astype(np.int64), h - 2)
    fu, fv = uc - u0, vc - v0
    top = img[v0, u0] * (1 - fu) + img[v0, u0 + 1] * fu
    bot = img[v0 + 1, u0] * (1 - fu) + img[v0 + 1, u0 + 1] * fu
    return top * (1 - fv) + bot * fv, inb


def _parabola_peak(left, centre, right):
    """Sub-sample offset of a peak from three equally spaced samples, in [-0.5, 0.5]."""
    denom = left - 2.0 * centre + right
    if denom >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


class AlignmentAccumulator:
    """Depth-agreement counts per (range bin, image shift), summed over frames.

    See the module docstring. Memory is bins x (2*max_shift+1)^2 counters, no
    matter how many frames are added.
    """

    def __init__(self, bins=DEFAULT_RANGE_BINS, max_shift=15, rel_tol=0.01, abs_tol=0.05):
        self.bins = np.asarray(bins, dtype=np.float64)
        self.max_shift = int(max_shift)
        self.rel_tol = float(rel_tol)
        self.abs_tol = float(abs_tol)
        self.reset()
        self._shifts = [(du, dv) for dv in range(-self.max_shift, self.max_shift + 1)
                        for du in range(-self.max_shift, self.max_shift + 1)]

    def reset(self):
        n = 2 * self.max_shift + 1
        nb = len(self.bins) - 1
        self.counts = np.zeros((nb, n, n), dtype=np.int64)   # [bin, dv, du]
        self.points = np.zeros(nb, dtype=np.int64)
        self.frames = 0

    def merge(self, other):
        """Add another accumulator's counts (same bins and max_shift) into this one."""
        self.counts += other.counts
        self.points += other.points
        self.frames += other.frames

    def add(self, uv, z, ranges, depth):
        """One frame: uv (N,2) projected pixels, z (N,) optical depth, ranges (N,)."""
        b = np.searchsorted(self.bins, ranges, side='right') - 1
        keep = (b >= 0) & (b < len(self.bins) - 1)
        uv, z, b = uv[keep], z[keep], b[keep]
        nb = len(self.bins) - 1
        self.points += np.bincount(b, minlength=nb)
        tol = np.maximum(self.abs_tol, self.rel_tol * z)
        m = self.max_shift
        for du, dv in self._shifts:
            d, inb = sample_bilinear(depth, uv[:, 0] + du, uv[:, 1] + dv)
            match = inb & np.isfinite(d) & (np.abs(d - z) <= tol)
            self.counts[:, dv + m, du + m] += np.bincount(b[match], minlength=nb)
        self.frames += 1

    def results(self):
        """Per bin: error vector (du, dv) in px = where points land minus where they should."""
        m = self.max_shift
        dv_grid, du_grid = np.mgrid[-m:m + 1, -m:m + 1]
        norm = np.hypot(du_grid, dv_grid)
        rows = []
        for i in range(len(self.bins) - 1):
            c = self.counts[i].astype(np.float64)
            n = int(self.points[i])
            row = {'lo': float(self.bins[i]), 'hi': float(self.bins[i + 1]), 'points': n,
                   'du': float('nan'), 'dv': float('nan'), 'err_px': float('nan'),
                   'match_zero_pct': float('nan'), 'match_best_pct': float('nan'),
                   'at_edge': False}
            if n > 0 and c.max() > 0:
                # Ties go to the smallest shift: an unobservable direction reads 0, not noise.
                key = c - 1e-6 * norm
                iv, iu = np.unravel_index(np.argmax(key), c.shape)
                su, sv = float(iu - m), float(iv - m)
                if 0 < iu < 2 * m:
                    su += _parabola_peak(c[iv, iu - 1], c[iv, iu], c[iv, iu + 1])
                if 0 < iv < 2 * m:
                    sv += _parabola_peak(c[iv - 1, iu], c[iv, iu], c[iv + 1, iu])
                row.update(du=-su, dv=-sv, err_px=float(np.hypot(su, sv)),
                           match_zero_pct=100.0 * c[m, m] / n,
                           match_best_pct=100.0 * c[iv, iu] / n,
                           at_edge=bool(iu in (0, 2 * m) or iv in (0, 2 * m)))
            rows.append(row)
        return rows


def format_alignment(rows, title):
    lines = [title,
             '  range [m]    points   error px (du, dv)   |err|   depth-match @0 / @best']
    for r in rows:
        flag = '  <- at search edge, error may be larger' if r['at_edge'] else ''
        lines.append('  %5.0f-%-5.0f %8d   (%+6.2f, %+6.2f)  %6.2f      %5.1f%% / %5.1f%%%s'
                     % (r['lo'], r['hi'], r['points'], r['du'], r['dv'], r['err_px'],
                        r['match_zero_pct'], r['match_best_pct'], flag))
    return '\n'.join(lines)
