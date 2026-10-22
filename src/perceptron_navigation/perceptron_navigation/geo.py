"""GPS <-> local metres, and following a route by sub-goals (no ROS).

Local tangent plane around a datum: x east, y north (ENU), equirectangular, which is what the
CARLA bridge's GNSS model uses (perceptron_carla/sim_sensors.offset_latlon) and is well under a
centimetre off over a few kilometres. On the rover the same holds for navsat_transform's map
frame when its datum is set: map origin = datum, axes ENU.
"""

import math

import numpy as np

EARTH_M_PER_DEG = 111320.0


def latlon_to_enu(lat, lon, lat0, lon0):
    """(lat, lon) -> (east, north) metres from the datum (lat0, lon0)."""
    return ((lon - lon0) * EARTH_M_PER_DEG * math.cos(math.radians(lat0)),
            (lat - lat0) * EARTH_M_PER_DEG)


def enu_to_latlon(east, north, lat0, lon0):
    return (lat0 + north / EARTH_M_PER_DEG,
            lon0 + east / (EARTH_M_PER_DEG * math.cos(math.radians(lat0))))


class Route:
    """A polyline the robot follows by sub-goals: Nav2's costmaps are rolling windows a few
    tens of metres wide, so a goal 500 m away is handed over as a carrot `spacing` metres
    ahead along the route, moved on whenever the robot gets within `advance` of it."""

    def __init__(self, points, spacing=30.0, advance=12.0):
        p = np.asarray(points, dtype=float)
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
        self.p = p[keep]
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.s = np.r_[0.0, np.cumsum(seg)]
        self.length = float(self.s[-1])
        self.spacing, self.advance = float(spacing), float(advance)
        self.subgoal_s = None

    def progress(self, xy):
        """Arc length of the route point nearest to xy."""
        if len(self.p) < 2:
            return 0.0
        a, b = self.p[:-1], self.p[1:]
        d = b - a
        L2 = np.maximum((d ** 2).sum(axis=1), 1e-12)
        t = np.clip(((np.asarray(xy) - a) * d).sum(axis=1) / L2, 0.0, 1.0)
        q = a + d * t[:, None]
        k = int(np.argmin(np.linalg.norm(q - xy, axis=1)))
        return float(self.s[k] + t[k] * math.sqrt(L2[k]))

    def at(self, s):
        """(x, y, yaw) at arc length s; yaw along the route."""
        s = min(max(s, 0.0), self.length)
        k = min(int(np.searchsorted(self.s, s, side='right')) - 1, len(self.p) - 2)
        k = max(k, 0)
        a, b = self.p[k], self.p[k + 1]
        seg = self.s[k + 1] - self.s[k]
        t = (s - self.s[k]) / seg if seg > 0 else 0.0
        x, y = a + (b - a) * t
        return float(x), float(y), math.atan2(b[1] - a[1], b[0] - a[0])

    def is_final(self):
        return self.subgoal_s is not None and self.subgoal_s >= self.length - 1e-6

    def next_subgoal(self, xy):
        """New sub-goal (x, y, yaw) when one is due, else None. Due: none sent yet, or the
        robot is within `advance` of the current one and it is not the end of the route."""
        if self.subgoal_s is not None:
            if self.is_final():
                return None
            gx, gy, _ = self.at(self.subgoal_s)
            if math.hypot(xy[0] - gx, xy[1] - gy) > self.advance:
                return None
        s = min(self.progress(xy) + self.spacing, self.length)
        if self.length - s < 0.5 * self.spacing:      # no stub of a last leg
            s = self.length
        self.subgoal_s = s
        return self.at(s)


class RoadGraph:
    """Shortest road route between two points, over lane-centre polylines.

    Undirected on purpose (Stage 11 runs without traffic rules: any lane, either way).
    Consecutive points of a polyline are joined, and each polyline end is joined to the
    nearest point of every other polyline within `join` metres, which is how lanes meet at
    junctions. Each point is also joined to the nearest point of every other lane within
    `lane_change` metres, at `lane_change_cost` extra: without those, turning onto a street in
    the "wrong" lane meant driving to the next junction and back (a 60 m detour in Town10).
    Edge weight = length (+ cost). Routes start and end with straight legs from the robot and
    to the goal to their nearest graph points.
    """

    def __init__(self, lanes, join=1.5, lane_change=4.0, lane_change_cost=5.0):
        from scipy.sparse import coo_matrix
        from scipy.spatial import cKDTree
        pts, lane_of, ends, rows, cols = [], [], [], [], []
        for k, poly in enumerate(lanes):
            if len(poly) < 2:
                continue
            i0 = len(pts)
            pts.extend(poly)
            lane_of.extend([k] * len(poly))
            rows.extend(range(i0, i0 + len(poly) - 1))
            cols.extend(range(i0 + 1, i0 + len(poly)))
            ends.extend([i0, i0 + len(poly) - 1])
        self.p = np.asarray(pts, dtype=float).reshape(-1, 2)
        lane_of = np.asarray(lane_of)
        for e in ends:
            d = np.linalg.norm(self.p - self.p[e], axis=1)
            near = np.flatnonzero((d < join) & (lane_of != lane_of[e]))
            best = {}
            for j in near:
                if lane_of[j] not in best or d[j] < d[best[lane_of[j]]]:
                    best[lane_of[j]] = j
            for j in best.values():
                rows.append(e)
                cols.append(j)
        extra = [0.0] * len(rows)
        if lane_change > 0 and len(self.p):
            best = {}
            for i, j in cKDTree(self.p).query_pairs(lane_change):
                if lane_of[i] == lane_of[j]:
                    continue
                for a, b in ((i, j), (j, i)):          # a -> nearest point of b's lane
                    key = (a, lane_of[b])
                    d = float(np.linalg.norm(self.p[a] - self.p[b]))
                    if key not in best or d < best[key][1]:
                        best[key] = (b, d)
            for (a, _), (b, _) in best.items():
                rows.append(a)
                cols.append(b)
                extra.append(lane_change_cost)
        rows, cols, extra = np.asarray(rows), np.asarray(cols), np.asarray(extra)
        w = np.maximum(np.linalg.norm(self.p[rows] - self.p[cols], axis=1), 1e-3) + extra
        n = len(self.p)
        self.adj = coo_matrix((np.r_[w, w], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()

    def nearest(self, xy):
        d = np.linalg.norm(self.p - np.asarray(xy, dtype=float), axis=1)
        k = int(np.argmin(d))
        return k, float(d[k])

    def route(self, start_xy, goal_xy):
        """[start, ...graph points..., goal] or None when the two are not connected."""
        from scipy.sparse.csgraph import dijkstra
        s, _ = self.nearest(start_xy)
        g, _ = self.nearest(goal_xy)
        dist, pred = dijkstra(self.adj, directed=False, indices=s, return_predecessors=True)
        if not np.isfinite(dist[g]):
            return None
        chain = [g]
        while chain[-1] != s:
            chain.append(int(pred[chain[-1]]))
        chain.reverse()
        return [tuple(map(float, start_xy))] + [tuple(self.p[i]) for i in chain] + [tuple(map(float, goal_xy))]
