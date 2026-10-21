"""Tests for the lane / ego RViz markers (viz_markers.py), on plain geometry: no CARLA."""

from visualization_msgs.msg import Marker

from perceptron_carla import viz_markers as vm


def straight_lane(n=5, left=(('broken',), 'white'), right='edge', junction=False):
    return {'centre': [(float(k), 0.0, 0.0) for k in range(n)], 'right': [(0.0, -1.0)] * n,
            'width': [3.0] * n, 'left_mark': [left] * n, 'right_mark': [right] * n, 'junction': junction}


def by_ns(arr):
    return {m.ns: m for m in arr.markers}


def test_lane_lines_offsets_and_dashes():
    m = by_ns(vm.lane_marker_array([straight_lane()]))
    assert len(m['lane_centre'].points) == 2 * 4
    edge = m['road_edge'].points
    assert len(edge) == 2 * 4 and all(abs(p.y + 1.5) < 1e-9 for p in edge)     # right = -y, w/2
    white = m['marking_white'].points
    assert len(white) == 2 * 2 and all(abs(p.y - 1.5) < 1e-9 for p in white)   # broken: every other
    assert all(abs(p.z - 0.03) < 1e-9 for p in white)


def test_double_line_and_junction():
    lane = straight_lane(3, left=(('solid', 'solid'), 'yellow'), right=None, junction=True)
    m = by_ns(vm.lane_marker_array([lane]))
    ys = sorted({round(p.y, 3) for p in m['marking_yellow'].points})
    assert ys == [1.44, 1.56]
    assert 'lane_centre_junction' in m and 'lane_centre' not in m and 'road_edge' not in m


def test_ego_marker_follows_base_footprint():
    arr = vm.ego_marker_array([-0.5, 2.0, -0.75, 0.75, 0.0, 1.4],
                              [(1.6, 0.6, 0.25), (1.6, -0.6, 0.25), (0.0, 0.6, 0.25), (0.0, -0.6, 0.25)],
                              rig={'x': 2.1, 'camera_z': 0.42, 'lidar_z': 0.415})
    assert all(m.frame_locked and m.header.frame_id == 'base_footprint' for m in arr.markers)
    assert len({m.id for m in arr.markers}) == len(arr.markers)
    assert sum(m.type == Marker.CYLINDER for m in arr.markers) == 4 + 1        # wheels + lidar
    top = max(m.pose.position.z + m.scale.z / 2 for m in arr.markers if m.type != Marker.ARROW)
    assert abs(top - 1.4) < 1e-6
