"""Rolling 3D-aware costmap with a dynamic risk layer (roadmap Stage 8), no ROS.

    Layer 3  dynamic risk   predicted paths of moving objects (Stage 7), weighted by
                            when the robot would get there
    Layer 2  inflation      robot radius + exponential safety margin around obstacles
    Layer 1  3D voxels      ground-free lidar obstacle points (Stage 4), with free-space
                            clearing along each lidar ray and time decay
             base           ground is free space

REPRESENTATION. A grid of nx x ny cells (res metres) in odom, kept centred on the robot
(it shifts by whole cells, so cell boundaries never move -- no resampling smear). Each
column holds nz voxels of z_res (5 cm); z is height above the ground plane under the robot
(Stage 4 works in base_footprint, whose z = 0 is that plane), so a slope does not turn
the road into a wall. 2.5D: good for a ground robot; no tunnels or bridges.

LAYER 1 -- VOXELS. Per voxel an occupancy log-odds L:
  * every frame L decays by exp(-dt / decay_time): what is no longer confirmed fades,
    which bounds how long a stale obstacle can block the robot (persistence)
  * every lidar ray (to a ground point or an obstacle point) lowers L in the voxels it
    passes through (l_miss): a car that drove off is cleared as soon as the lidar sees
    through the space it occupied, not only when it has decayed. Rays to the GROUND
    matter as much as rays to obstacles: from a 0.4 m sensor only they sweep the bottom
    of the space an object left (rays to a wall behind pass over it)
  * obstacle points raise L in their voxel (l_hit)
  * a column whose ground the lidar saw this frame, with no obstacle point in it, gets
    l_miss at every height (see _clear_ground_columns: a low sensor cannot clear a
    vacated car-height space by rays alone)
  * points within a moving object's footprint are NOT added: a walking person would
    otherwise paint a trail of permanent obstacles behind them; they belong to layer 3
  Projection to 2D: a column is lethal if any voxel with height in
  [min_obstacle_height, max_obstacle_height] has L > occupied_threshold. Below the
  minimum is a bump the wheels roll over; above the maximum is an overhang the robot
  passes under.

LAYER 2 -- INFLATION (the Nav2 cost convention, 0-254):
    254 lethal  |  253 within inscribed_radius (the robot's centre there means contact)
    252 * exp(-cost_scaling * (d - inscribed_radius)) out to inflation_radius
  from an exact Euclidean distance transform of the lethal cells.

LAYER 3 -- DYNAMIC RISK. For each moving predicted object, at every point tau of its
predicted path (0..2 s): a Gaussian with the predicted covariance at tau, widened by
(object radius + robot radius), times a timing weight
    max( exp(-tau / near_term),  exp(-(t_robot - tau)^2 / (2 sigma_t^2)) )
where t_robot = distance(cell, robot) / max(robot speed, nominal speed). The second term
is time-to-encounter: a cell on a pedestrian's path matters most if the robot would reach
it at about the moment the pedestrian does. The first keeps the next ~second of any
path costly regardless. The object's current footprint is lethal. Risk 0..1 maps to
cost 0..252, so it shapes a planner's path but is never confused with a static wall.

COMBINED = max(layer 2, layer 3); cells never observed (or not for unknown_timeout) are
reported as unknown.
"""

import math

import numpy as np
from scipy.ndimage import distance_transform_edt

LETHAL, INSCRIBED, MAX_NON_LETHAL = 254, 253, 252


class RollingCostmap:

    def __init__(self, size=40.0, res=0.2, z_res=0.05, z_max=2.0, min_obstacle_height=0.05,
                 max_obstacle_height=0.8, l_hit=0.85, l_miss=-0.4, l_min=-2.0, l_max=3.5,
                 occupied_threshold=0.6, decay_time=3.0, unknown_timeout=10.0,
                 inscribed_radius=0.21, inflation_radius=0.6, cost_scaling=2.5,
                 max_rays=3500, ray_stop=1.5):
        self.res, self.z_res = res, z_res
        self.nx = self.ny = int(round(size / res))
        self.nz = int(round(z_max / z_res))
        # the band starts on a voxel boundary at or above min_obstacle_height, so a bump
        # below it never shares a voxel with something that counts
        self.k_min = max(0, int(math.ceil(min_obstacle_height / z_res - 1e-9)))
        self.k_max = min(self.nz, int(math.ceil(max_obstacle_height / z_res)))
        self.l_hit, self.l_miss, self.l_min, self.l_max = l_hit, l_miss, l_min, l_max
        self.thr = occupied_threshold
        self.decay_time, self.unknown_timeout = decay_time, unknown_timeout
        self.inscribed, self.inflation, self.scaling = inscribed_radius, inflation_radius, cost_scaling
        self.max_rays, self.ray_stop = max_rays, ray_stop
        self.L = np.zeros((self.nx, self.ny, self.nz), np.float32)
        self.seen = np.full((self.nx, self.ny), -np.inf)
        self.origin = None                    # odom xy of cell (0, 0)'s corner
        self.t = None
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ geometry
    def recenter(self, robot_xy):
        """Keep the robot near the centre, shifting by whole cells."""
        half = self.nx * self.res / 2.0
        want = np.floor((np.asarray(robot_xy, float) - half) / self.res) * self.res
        if self.origin is None:
            self.origin = want
            return
        shift = np.rint((want - self.origin) / self.res).astype(int)
        if np.all(np.abs(shift) < max(2, self.nx // 10)):
            return                            # small motion: no shift yet
        self.L = _shift(self.L, shift, 0.0)
        self.seen = _shift(self.seen, shift, -np.inf)
        self.origin = self.origin + shift * self.res

    def cell_of(self, xy):
        ij = np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(np.int64)
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < self.nx) & (ij[..., 1] >= 0) & (ij[..., 1] < self.ny)
        return ij, ok

    def cell_centres(self):
        x = self.origin[0] + (np.arange(self.nx) + 0.5) * self.res
        y = self.origin[1] + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(x, y, indexing='ij')

    # ------------------------------------------------------------------ layer 1
    def integrate(self, obstacles, ground, sensor, t, exclude=()):
        """obstacles, ground: (N,3) with x, y in odom and z = height above ground.
        sensor: (3,) lidar origin, same convention. exclude: [(xy, radius)] footprints of
        moving objects whose points must not enter the voxel layer."""
        if self.t is not None:
            dt = max(0.0, t - self.t)
            self.L *= math.exp(-dt / self.decay_time)
        self.t = t
        obstacles = np.asarray(obstacles, float).reshape(-1, 3)
        ground = np.asarray(ground, float).reshape(-1, 3)
        for xy, r in exclude:
            obstacles = obstacles[np.hypot(obstacles[:, 0] - xy[0], obstacles[:, 1] - xy[1]) > r]
        ends = np.vstack([obstacles, ground])
        if len(ends) > self.max_rays:
            ends = ends[self.rng.choice(len(ends), self.max_rays, replace=False)]
        self._clear_rays(np.asarray(sensor, float), ends, t)
        if len(obstacles):
            ij, ok = self.cell_of(obstacles[:, :2])
            k = np.floor(obstacles[:, 2] / self.z_res).astype(np.int64)
            ok &= (k >= 0) & (k < self.nz)
            # a[idx] += v applies each duplicate index once: one l_hit per voxel per frame
            flat = np.ravel_multi_index((ij[ok, 0], ij[ok, 1], k[ok]), self.L.shape)
            self.L.ravel()[flat] += self.l_hit
            self.seen[ij[ok, 0], ij[ok, 1]] = t
        self._clear_ground_columns(ground, obstacles)
        np.clip(self.L, self.l_min, self.l_max, out=self.L)

    def _clear_ground_columns(self, ground, obstacles):
        """A column where this frame's lidar saw the ground and no obstacle point is free.

        Ray clearing alone cannot do it from a low sensor: the ray to a road point arrives
        nearly flat, so it never passes back through the 0.2-1.4 m a car just vacated, and
        measured CARLA runs showed 91 % of false obstacles on the road -- moving-car trails.
        One l_miss per frame (an obstacle still there scores l_hit, which is larger).
        """
        if len(ground) == 0:
            return
        gij, gok = self.cell_of(ground[:, :2])
        g = np.zeros((self.nx, self.ny), bool)
        g[gij[gok, 0], gij[gok, 1]] = True
        if len(obstacles):
            oij, ook = self.cell_of(obstacles[:, :2])
            g[oij[ook, 0], oij[ook, 1]] = False
        self.L[g] += self.l_miss

    def _clear_rays(self, sensor, ends, t):
        if len(ends) == 0:
            return
        d = ends - sensor
        length = np.linalg.norm(d, axis=1)
        stop = np.maximum(length - self.ray_stop * self.res, 0.0)   # never clear the hit voxel
        step = self.res * 0.75              # still lands in every cell a ray crosses
        n = np.ceil(stop / step).astype(np.int64)
        keep = n > 0
        if not keep.any():
            return
        d, length, n = d[keep], length[keep], n[keep]
        ray = np.repeat(np.arange(len(n)), n)
        first = np.cumsum(n) - n
        s = (np.arange(n.sum()) - np.repeat(first, n)) * step          # distance along each ray
        p = sensor + d[ray] * (s / length[ray])[:, None]
        ij, ok = self.cell_of(p[:, :2])
        k = np.floor(p[:, 2] / self.z_res).astype(np.int64)
        self.seen[ij[ok, 0], ij[ok, 1]] = t                             # observed columns
        ok &= (k >= 0) & (k < self.nz)
        # no np.unique: a[idx] += v already applies each duplicate once (one l_miss per voxel)
        flat = np.ravel_multi_index((ij[ok, 0], ij[ok, 1], k[ok]), self.L.shape)
        self.L.ravel()[flat] += self.l_miss

    def occupied(self):
        return self.L > self.thr

    def lethal(self):
        return self.occupied()[:, :, self.k_min:self.k_max].any(axis=2)

    def known(self):
        """Columns observed (by a ray or a hit) within unknown_timeout."""
        if self.t is None:
            return np.zeros((self.nx, self.ny), bool)
        return (self.t - self.seen) <= self.unknown_timeout

    # ------------------------------------------------------------------ layer 2
    def inflate(self, lethal):
        cost = np.zeros(lethal.shape, np.uint8)
        if not lethal.any():
            return cost
        dist = distance_transform_edt(~lethal) * self.res
        ring = (dist > self.inscribed) & (dist <= self.inflation)
        cost[ring] = np.minimum(MAX_NON_LETHAL, np.round(
            MAX_NON_LETHAL * np.exp(-self.scaling * (dist[ring] - self.inscribed)))).astype(np.uint8)
        cost[dist <= self.inscribed] = INSCRIBED
        cost[lethal] = LETHAL
        return cost

    # ------------------------------------------------------------------ layer 3
    def dynamic_risk(self, objects, robot_xy, robot_speed, nominal_speed=1.0, near_term=0.8, sigma_t=0.75):
        """objects: [dict(path (N,2), path_dt, horizons, covs (H,2,2), radius)] for MOVING
        predictions. Returns (risk 0..1 float grid, lethal footprint mask)."""
        risk = np.zeros((self.nx, self.ny), np.float32)
        lethal = np.zeros((self.nx, self.ny), bool)
        if not objects:
            return risk, lethal
        cx, cy = self.cell_centres()
        v_plan = max(float(robot_speed), nominal_speed)
        t_robot = np.hypot(cx - robot_xy[0], cy - robot_xy[1]) / v_plan
        for ob in objects:
            path = np.asarray(ob['path'], float)
            hz = np.asarray(ob['horizons'], float)
            covs = np.asarray(ob['covs'], float)
            spread = (ob['radius'] + self.inscribed) / 2.0          # 2 sigma covers both radii
            for m, mu in enumerate(path):
                tau = m * ob['path_dt']
                S = _interp_cov(tau, hz, covs) + np.eye(2) * spread ** 2
                half = 3.0 * math.sqrt(max(np.linalg.eigvalsh(S).max(), 1e-6))
                (i0, j0), _ = self.cell_of(mu - half)
                (i1, j1), _ = self.cell_of(mu + half)
                i0, j0 = max(i0, 0), max(j0, 0)
                i1, j1 = min(i1 + 1, self.nx), min(j1 + 1, self.ny)
                if i0 >= i1 or j0 >= j1:
                    continue
                dx = cx[i0:i1, j0:j1] - mu[0]
                dy = cy[i0:i1, j0:j1] - mu[1]
                Si = np.linalg.inv(S)
                m2 = Si[0, 0] * dx * dx + 2 * Si[0, 1] * dx * dy + Si[1, 1] * dy * dy
                timing = np.maximum(math.exp(-tau / near_term),
                                    np.exp(-(t_robot[i0:i1, j0:j1] - tau) ** 2 / (2 * sigma_t ** 2)))
                np.maximum(risk[i0:i1, j0:j1], np.exp(-0.5 * m2) * timing, out=risk[i0:i1, j0:j1])
                if m == 0:
                    lethal[i0:i1, j0:j1] |= np.hypot(dx, dy) <= ob['radius']
        return risk, lethal


def risk_to_cost(risk, lethal):
    cost = np.round(np.clip(risk, 0.0, 1.0) * MAX_NON_LETHAL).astype(np.uint8)
    cost[lethal] = LETHAL
    return cost


def to_occupancy(cost, known=None):
    """Nav2 cost 0..254 -> OccupancyGrid 0..100 (-1 unknown), int8, row-major y then x."""
    occ = np.round(cost.astype(np.float32) * 100.0 / LETHAL).astype(np.int8)
    if known is not None:
        occ[~known & (cost == 0)] = -1
    return occ.T.copy()                          # OccupancyGrid rows run along y


def _interp_cov(tau, horizons, covs):
    if tau <= horizons[0]:
        return covs[0]
    if tau >= horizons[-1]:
        return covs[-1]
    k = int(np.searchsorted(horizons, tau))
    a = (tau - horizons[k - 1]) / (horizons[k] - horizons[k - 1])
    return (1 - a) * covs[k - 1] + a * covs[k]


def _shift(arr, shift, fill):
    """Content moves by -shift cells (the window moved by +shift); vacated cells = fill."""
    out = np.full_like(arr, fill)
    sx, sy = int(shift[0]), int(shift[1])
    nx, ny = arr.shape[0], arr.shape[1]
    if abs(sx) >= nx or abs(sy) >= ny:
        return out
    src_x = slice(max(sx, 0), nx + min(sx, 0))
    dst_x = slice(max(-sx, 0), nx + min(-sx, 0))
    src_y = slice(max(sy, 0), ny + min(sy, 0))
    dst_y = slice(max(-sy, 0), ny + min(-sy, 0))
    out[dst_x, dst_y] = arr[src_x, src_y]
    return out
