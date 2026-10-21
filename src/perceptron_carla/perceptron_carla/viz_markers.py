"""RViz markers for the CARLA world: lane lines from the OpenDRIVE map, and the ego car.

The lane lines are ground truth from CARLA's map (sim only: the rover has no HD map); the
camera-based equivalent is perceptron_perception/road_segmentation_node. Geometry is
collected once into plain lists (lane_polylines, needs a carla.Map) and turned into
markers by functions that need no CARLA, so they are testable.

CARLA is left-handed (y right); ROS is y left, so every y is negated here.
"""

import math

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# Marking styles, by name of carla.LaneMarkingType: one entry per painted line, inner first
STYLES = {
    'Solid': ('solid',), 'Broken': ('broken',), 'SolidSolid': ('solid', 'solid'),
    'SolidBroken': ('solid', 'broken'), 'BrokenSolid': ('broken', 'solid'),
    'BrokenBroken': ('broken', 'broken'), 'BottsDots': ('broken',),
}
COLOURS = {
    'white': (0.95, 0.95, 0.95, 1.0), 'yellow': (1.0, 0.8, 0.1, 1.0),
    'edge': (0.75, 0.75, 0.8, 1.0), 'centre': (0.2, 0.8, 1.0, 0.55), 'junction': (0.2, 0.8, 1.0, 0.25),
}
DOUBLE_GAP = 0.12        # m between the two lines of a double marking


def lane_polylines(carla_map, step=1.5):
    """Driving lanes of a carla.Map -> list of dicts with ROS-frame geometry:
    centre [(x, y, z)], right [(x, y)] unit vectors, width [m], left/right marking
    (style, colour) or 'edge' or None per sample, and junction flag."""
    import carla
    out = []
    for start, _ in carla_map.get_topology():
        if start.lane_type != carla.LaneType.Driving:
            continue
        wps = [start] + start.next_until_lane_end(step)
        lane = {'centre': [], 'right': [], 'width': [], 'left_mark': [], 'right_mark': [],
                'junction': bool(start.is_junction)}
        for wp in wps:
            loc, rv = wp.transform.location, wp.transform.get_right_vector()
            n = math.hypot(rv.x, rv.y) or 1.0
            lane['centre'].append((loc.x, -loc.y, loc.z))
            lane['right'].append((rv.x / n, -rv.y / n))
            lane['width'].append(wp.lane_width)
            lane['left_mark'].append(_marking(wp.left_lane_marking, wp.get_left_lane()))
            lane['right_mark'].append(_marking(wp.right_lane_marking, wp.get_right_lane()))
        if len(lane['centre']) >= 2:
            out.append(lane)
    return out


def _marking(m, neighbour):
    """carla.LaneMarking -> (style tuple, colour name), 'edge' (no paint, road ends), or None."""
    import carla
    name = str(m.type).split('.')[-1]
    if name in STYLES:
        colour = 'yellow' if m.color == carla.LaneMarkingColor.Yellow else 'white'
        return (STYLES[name], colour)
    if neighbour is None or neighbour.lane_type not in (carla.LaneType.Driving, carla.LaneType.Bidirectional):
        return 'edge'
    return None


def _colour(name):
    return ColorRGBA(**dict(zip('rgba', COLOURS[name])))


def _line(ns, mid, frame, width, colour):
    m = Marker()
    m.header.frame_id = frame
    m.ns, m.id, m.type, m.action = ns, mid, Marker.LINE_LIST, Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = width
    m.color = _colour(colour)
    return m


def lane_marker_array(lanes, frame='carla_world', z_lift=0.03):
    """Lane centre lines (thin cyan; fainter inside junctions), painted markings (white /
    yellow, solid / broken, doubles as two lines) and unpainted road edges (grey)."""
    lines = {
        'centre': _line('lane_centre', 0, frame, 0.06, 'centre'),
        'junction': _line('lane_centre_junction', 1, frame, 0.04, 'junction'),
        'white': _line('marking_white', 2, frame, 0.12, 'white'),
        'yellow': _line('marking_yellow', 3, frame, 0.12, 'yellow'),
        'edge': _line('road_edge', 4, frame, 0.08, 'edge'),
    }

    def seg(key, a, b):
        lines[key].points += [Point(x=a[0], y=a[1], z=a[2]), Point(x=b[0], y=b[1], z=b[2])]

    for lane in lanes:
        c = lane['centre']
        for k in range(len(c) - 1):
            a, b = (c[k][0], c[k][1], c[k][2] + z_lift), (c[k + 1][0], c[k + 1][1], c[k + 1][2] + z_lift)
            seg('junction' if lane['junction'] else 'centre', a, b)
            for side, sign in (('left_mark', -1.0), ('right_mark', 1.0)):
                mark = lane[side][k]
                if mark is None:
                    continue
                if mark == 'edge':
                    styles, colour = ('solid',), 'edge'
                else:
                    styles, colour = mark
                for q, style in enumerate(styles):
                    if style == 'broken' and k % 2:
                        continue                       # 1 step painted, 1 step gap
                    off = sign * (lane['width'][k] / 2.0 + q * DOUBLE_GAP - (len(styles) - 1) * DOUBLE_GAP / 2.0)
                    pa = _offset(a, lane['right'][k], off)
                    pb = _offset(b, lane['right'][k + 1], off)
                    seg(colour, pa, pb)
    return MarkerArray(markers=[m for m in lines.values() if m.points])


def _offset(p, right, d):
    return (p[0] + right[0] * d, p[1] + right[1] * d, p[2])


def ego_marker_array(host_box, wheels, rig=None, frame='base_footprint'):
    """The host car as primitives, locked to base_footprint so it follows TF.

    host_box: [x0, x1, y0, y1, z0, z1] in base_footprint (bridge_info's host box)
    wheels:   [(x, y, radius)] in base_footprint
    rig:      the rover sensor rig {'x', 'camera_z', 'lidar_z'} in base_footprint, or None
    """
    x0, x1, y0, y1, z0, z1 = host_box
    L, W, H = x1 - x0, y1 - y0, z1 - z0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    r = max([w[2] for w in wheels], default=0.25)
    out = []

    def add(kind, pos, scale, rgba, quat=(0.0, 0.0, 0.0, 1.0)):
        m = Marker()
        m.header.frame_id = frame
        m.ns, m.id, m.type, m.action = 'ego', len(out), kind, Marker.ADD
        m.frame_locked = True
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = quat
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        out.append(m)

    body_top = z0 + r + 0.45 * (H - r)
    add(Marker.CUBE, (cx, cy, (z0 + r * 0.6 + body_top) / 2.0), (L, W, body_top - z0 - r * 0.6),
        (0.15, 0.55, 0.75, 0.9))                                                   # lower body
    add(Marker.SPHERE, (cx - 0.05 * L, cy, body_top), (0.9 * L, 0.92 * W, 2.0 * (z1 - body_top)),
        (0.6, 0.85, 1.0, 0.45))                                                    # glass bubble
    half = math.sqrt(0.5)
    for (wx, wy, wr) in wheels:
        add(Marker.CYLINDER, (wx, wy, wr), (2 * wr, 2 * wr, 0.18), (0.08, 0.08, 0.08, 1.0),
            (half, 0.0, 0.0, half))                                                # axis along y
    for side in (-1.0, 1.0):
        add(Marker.CUBE, (x1 - 0.02, cy + side * 0.32 * W, body_top - 0.1), (0.05, 0.18, 0.08),
            (1.0, 0.95, 0.5, 1.0))                                                 # headlights
    add(Marker.ARROW, (cx - 0.3 * L, cy, z1 + 0.15), (0.6 * L, 0.12, 0.12), (1.0, 1.0, 1.0, 0.8))   # heading
    if rig is not None:
        top = max(rig['camera_z'], rig['lidar_z'])
        add(Marker.CUBE, (rig['x'], 0.0, top / 2.0), (0.05, 0.05, top), (0.4, 0.4, 0.45, 1.0))    # mast
        add(Marker.CUBE, (rig['x'] + 0.04, 0.0, rig['camera_z']), (0.06, 0.16, 0.06),
            (0.9, 0.3, 0.2, 1.0))                                                  # camera
        add(Marker.CYLINDER, (rig['x'], 0.0, rig['lidar_z']), (0.1, 0.1, 0.07),
            (0.15, 0.15, 0.15, 1.0))                                               # lidar
    return MarkerArray(markers=out)
