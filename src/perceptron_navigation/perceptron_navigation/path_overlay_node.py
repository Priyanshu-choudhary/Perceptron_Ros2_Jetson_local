#!/usr/bin/env python3
"""Project the Nav2 plan into camera pixels and ship them to the Jetson to draw.

    /plan (map)  -> TF -> camera_optical_link -> cv2.projectPoints -> (u,v)
                 --ZMQ b"overlay" PUSH :5558--> jetson_path_overlay.py
                                                 draws on the live frame
                                                 H.265 -> udpsink -> operator

WHY THE PIXELS ARE COMPUTED HERE AND DRAWN THERE

The camera never joins the ROS graph. The Jetson owns /dev/video0 and hands the
frames straight to nvv4l2h265enc, because an uncompressed 1280x720 stream is
~41 MB/s and perceptron_robot_description/README.md records what that does to
node discovery on a 4 GB WSL host. So the frame cannot come to the geometry.

The geometry can go to the frame, though, and it is tiny: a decimated path is
~250 integer pixel pairs, about 1.2 kB, or 18 kB/s at 15 Hz. That also keeps
every piece of knowledge on the side that already has it -- TF, the intrinsics
and the Nav2 topics stay here, and the Jetson receives a list of shapes with no
idea that ROS exists. The wire format below is deliberately dumb for that
reason: polygons, polylines, circles, labels. Teaching the renderer about paths
would put robot knowledge on the far side of a link it cannot introspect.

THE PROJECTION ITSELF lives in camera_projection.py, shared with the lidar
overlay, including why the cull is Z > min_z AND r <= max_radius rather than
just Z > 0. Only the path-specific parts (runs, ribbons, the wire) are here.

WHY THE TF LOOKUP ASKS FOR THE LATEST TRANSFORM

Ideally we would look up at the frame's capture time. We cannot: the frame is
on the other side of a UDP video link with no shared clock, and the Jetson's
time.time() is not disciplined against this host's -- jetson_bridge_node's
JetsonClock exists solely to paper over that. Asking for the latest transform
is what aruco_detector_node already does and is the honest option here. The
cost is that while the robot is moving the overlay lags the picture by the
video pipeline's latency and appears to swim slightly. That is cosmetic and
expected; it is not a projection bug.

NOTHING HERE STEERS THE ROBOT. If the link dies the operator loses a drawing.
"""

import math
import os
import time

import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

import msgpack
import zmq

from perceptron_navigation import camera_projection as cp

WIRE_VERSION = 1


def runs_of_true(mask):
    """Yield (start, stop) slices of consecutive True in a boolean array.

    A polyline must only be drawn between two points that both survived the
    cull. Joining across a gap draws a chord through whatever the cull was
    protecting against, which is how the phantom stripes appear.
    """
    if mask.size == 0:
        return
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    start = 0
    for e in edges:
        if mask[start]:
            yield start, e + 1
        start = e + 1
    if mask[start]:
        yield start, mask.size


def path_to_xyz(path_msg):
    """nav_msgs/Path -> (N, 3) float64. Keeps the pose z rather than assuming 0."""
    n = len(path_msg.poses)
    out = np.empty((n, 3), dtype=np.float64)
    for i, ps in enumerate(path_msg.poses):
        p = ps.pose.position
        out[i] = (p.x, p.y, p.z)
    return out


def ribbon_edges(pts, width):
    """Offset a centreline by +/- width/2 in the ground plane.

    The offset is taken in the world, not in the image. Offsetting projected
    pixels by a constant would draw a ribbon of constant screen width, which
    reads as a flat sticker; offsetting in metres first lets perspective narrow
    it with distance, which is what makes it look like it lies on the floor.
    """
    if len(pts) < 2:
        return None, None
    tangent = np.gradient(pts[:, :2], axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    # A stationary robot can leave duplicate poses in the plan, giving a
    # zero-length tangent. Carry the previous direction through rather than
    # dividing by zero and spraying NaN into the polygon.
    bad = norms[:, 0] < 1e-9
    if bad.any():
        good = np.flatnonzero(~bad)
        if good.size == 0:
            return None, None
        idx = np.clip(np.searchsorted(good, np.arange(len(pts))), 0, good.size - 1)
        tangent = tangent[good[idx]]
        norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / norms
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    half = np.zeros_like(pts)
    half[:, :2] = normal * (width / 2.0)
    return pts + half, pts - half


class PathOverlayNode(Node):

    def __init__(self):
        super().__init__('path_overlay_node')

        default_jetson_ip = os.environ.get('JETSON_IP', '192.168.1.7')

        self.declare_parameter('jetson_ip', default_jetson_ip)
        self.declare_parameter('overlay_port', 5558)

        self.declare_parameter('global_path_topic', '/plan')
        self.declare_parameter('local_path_topic', '/local_plan')

        self.declare_parameter('camera_optical_frame', 'camera_optical_link')

        # Same measured values as aruco_localization.yaml, at the 1280x720 they
        # were taken at. The Jetson rescales to whatever it is actually
        # encoding, so these stay at calibration resolution no matter what the
        # video pipeline is set to.
        self.declare_parameter('camera_matrix', [
            410.9212916385953, 0.0, 634.8042096493966,
            0.0, 411.4395116912116, 347.58359871233904,
            0.0, 0.0, 1.0])
        self.declare_parameter('dist_coeffs', [
            0.030287865182229395, -0.002860707267771121,
            -0.004223044189411946, -0.0003374289867825264,
            -0.0003820853552867496])
        self.declare_parameter('calibration_width', 1280)
        self.declare_parameter('calibration_height', 720)

        self.declare_parameter('publish_rate', 15.0)
        self.declare_parameter('path_timeout', 2.0)

        # --- cull -----------------------------------------------------------
        self.declare_parameter('min_z', 0.15)
        # tan(67 deg). The image corner is r = 1.76 and the radial model stays
        # monotonic to about 2.4, so this admits the whole frame with margin
        # while stopping the fold-back described in the module docstring.
        self.declare_parameter('max_radius', 2.35)
        # Beyond this the path is within a few pixels of the horizon and adds
        # nothing but clutter and jitter.
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('max_points_per_line', 256)

        # --- style ----------------------------------------------------------
        # The chassis is 0.415 x 0.4526 m (nav2_params.yaml header), so this is
        # the real width. The ribbon is then a clearance gauge and not just
        # decoration: if it fits between two obstacles in the picture, so does
        # the robot.
        self.declare_parameter('ribbon_width', 0.4526)
        self.declare_parameter('ribbon_alpha', 0.30)
        self.declare_parameter('draw_ribbon', True)
        self.declare_parameter('draw_local_plan', True)
        self.declare_parameter('draw_goal', True)
        self.declare_parameter('global_rgb', [60, 220, 90])
        self.declare_parameter('local_rgb', [70, 160, 255])
        self.declare_parameter('goal_rgb', [255, 210, 40])

        p = self.get_parameter
        self.camera_optical_frame = p('camera_optical_frame').value
        self.calib_size = (int(p('calibration_width').value),
                           int(p('calibration_height').value))
        self.camera_matrix = np.array(p('camera_matrix').value,
                                      dtype=np.float64).reshape((3, 3))
        self.dist_coeffs = np.array(p('dist_coeffs').value, dtype=np.float64)

        self.path_timeout = float(p('path_timeout').value)
        self.min_z = float(p('min_z').value)
        self.max_radius = float(p('max_radius').value)
        self.max_range = float(p('max_range').value)
        self.max_points = int(p('max_points_per_line').value)

        self.ribbon_width = float(p('ribbon_width').value)
        self.ribbon_alpha = float(p('ribbon_alpha').value)
        self.draw_ribbon = bool(p('draw_ribbon').value)
        self.draw_local = bool(p('draw_local_plan').value)
        self.draw_goal = bool(p('draw_goal').value)
        self.global_rgb = [int(c) for c in p('global_rgb').value]
        self.local_rgb = [int(c) for c in p('local_rgb').value]
        self.goal_rgb = [int(c) for c in p('goal_rgb').value]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.global_path = None
        self.global_path_t = 0.0
        self.local_path = None
        self.local_path_t = 0.0

        self.global_path_topic = p('global_path_topic').value
        self.create_subscription(Path, self.global_path_topic,
                                 self._global_cb, 10)
        self.create_subscription(Path, p('local_path_topic').value,
                                 self._local_cb, 10)

        jetson_ip = p('jetson_ip').value
        port = int(p('overlay_port').value)
        self.zmq_context = zmq.Context()
        self.sock = self.zmq_context.socket(zmq.PUSH)
        # The renderer wants the newest drawing and nothing else. A deep queue
        # here would bank overlays during a Wi-Fi stall and then deliver a
        # burst of stale geometry, which is the same failure the bridge's
        # RCVHWM comment describes for telemetry: late data labelled as current
        # is worse than no data. Two buffers plus NOBLOCK drops instead.
        self.sock.setsockopt(zmq.SNDHWM, 2)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.overlay_url = f'tcp://{jetson_ip}:{port}'
        self.sock.connect(self.overlay_url)

        self.seq = 0
        self.sent = 0
        self.dropped = 0
        self.last_error = ''

        rate = float(p('publish_rate').value)
        self.create_timer(1.0 / max(rate, 1.0), self._tick)
        self.create_timer(10.0, self._report)

        self.get_logger().info(
            f'Path overlay up: projecting into {self.camera_optical_frame} at '
            f'{self.calib_size[0]}x{self.calib_size[1]}, pushing to {self.overlay_url}')

    # ---------------------------------------------------------------- inputs

    def _global_cb(self, msg):
        self.global_path = msg
        self.global_path_t = time.monotonic()

    def _local_cb(self, msg):
        self.local_path = msg
        self.local_path_t = time.monotonic()

    def _fresh(self, path, stamp):
        if path is None or not path.poses:
            return None
        if time.monotonic() - stamp > self.path_timeout:
            return None
        return path

    # ------------------------------------------------------------ projection

    def _lookup(self, source_frame):
        """4x4 camera_optical <- source_frame, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_optical_frame, source_frame, rclpy.time.Time())
        except Exception as exc:
            # Throttled rather than once-only: a transform that is missing for
            # ten minutes should still be saying so at minute ten, because the
            # operator looking at blank video needs to know which of the four
            # reasons for blankness this is.
            self.last_error = f'no tf {source_frame} -> {self.camera_optical_frame}'
            self.get_logger().warn(
                f'No transform {source_frame} -> {self.camera_optical_frame} '
                f'({exc}); overlay idle.', throttle_duration_sec=10.0)
            return None
        return cp.transform_to_matrix(tf)

    def _to_optical(self, M, pts):
        return cp.transform_points(M, pts)

    def _visible(self, pts_opt):
        return cp.visible_mask(pts_opt, self.min_z, self.max_radius, self.max_range)

    def _project(self, pts_opt):
        """Optical-frame points -> (N, 2) int pixel coords at calibration size."""
        return cp.project_pixels(pts_opt, self.camera_matrix, self.dist_coeffs)

    def _decimate(self, idx):
        """Stride a run down to max_points, always keeping both ends."""
        if len(idx) <= self.max_points:
            return idx
        step = int(math.ceil(len(idx) / self.max_points))
        kept = idx[::step]
        if kept[-1] != idx[-1]:
            kept = np.append(kept, idx[-1])
        return kept

    def _polylines_for(self, M, pts_world, rgb, width):
        """Cull, decimate and project one centreline into wire polylines."""
        pts_opt = self._to_optical(M, pts_world)
        mask = self._visible(pts_opt)
        out = []
        for a, b in runs_of_true(mask):
            if b - a < 2:
                continue
            idx = self._decimate(np.arange(a, b))
            uv = self._project(pts_opt[idx])
            out.append({'pts': uv.tolist(), 'rgb': rgb, 'w': width})
        return out

    def _ribbon_for(self, M, pts_world):
        """A translucent floor band of true robot width, as one polygon per run."""
        left, right = ribbon_edges(pts_world, self.ribbon_width)
        if left is None:
            return []
        l_opt = self._to_optical(M, left)
        r_opt = self._to_optical(M, right)
        c_opt = self._to_optical(M, pts_world)
        # All three edges must survive, or the band would be built from a
        # centreline the sides do not actually bracket.
        mask = self._visible(l_opt) & self._visible(r_opt) & self._visible(c_opt)
        out = []
        for a, b in runs_of_true(mask):
            if b - a < 2:
                continue
            idx = self._decimate(np.arange(a, b))
            lu = self._project(l_opt[idx])
            ru = self._project(r_opt[idx])
            ring = np.vstack([lu, ru[::-1]])
            out.append({'pts': ring.tolist(), 'rgb': self.global_rgb,
                        'alpha': self.ribbon_alpha})
        return out

    # ----------------------------------------------------------------- cycle

    def _tick(self):
        gpath = self._fresh(self.global_path, self.global_path_t)
        lpath = self._fresh(self.local_path, self.local_path_t) if self.draw_local else None

        payload = {
            'v': WIRE_VERSION,
            'seq': self.seq,
            'ref': list(self.calib_size),
            'polygons': [],
            'polylines': [],
            'circles': [],
            'labels': [],
            'hud': [],
        }
        self.seq += 1

        if gpath is None and lpath is None:
            # Distinguish "the planner is not running" from "the planner is
            # running and idle". count_publishers is the cheapest honest test:
            # with Nav2 down there is nobody on /plan at all, whereas an idle
            # Nav2 holds the publisher open between goals.
            if self.count_publishers(self.global_path_topic) == 0:
                payload['hud'].append('no plan - planner not running')
            else:
                payload['hud'].append('no plan - idle, send a goal')
            # Send the empty frame anyway. Silence would leave the renderer
            # holding the last drawing until its own timeout, so an overlay
            # from the previous goal would linger over a stopped robot.
            self._send(payload)
            return

        frame_id = (gpath or lpath).header.frame_id or 'map'
        M = self._lookup(frame_id)
        if M is None:
            payload['hud'].append(self.last_error or 'no tf')
            self._send(payload)
            return

        if gpath is not None:
            pts = path_to_xyz(gpath)
            if self.draw_ribbon:
                payload['polygons'].extend(self._ribbon_for(M, pts))
            payload['polylines'].extend(
                self._polylines_for(M, pts, self.global_rgb, 3))

            remaining = float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum())
            payload['hud'].append(f'plan {remaining:5.2f} m  {len(pts)} pts')

            if self.draw_goal:
                goal = pts[-1:].copy()
                g_opt = self._to_optical(M, goal)
                if self._visible(g_opt).all():
                    uv = self._project(g_opt)[0]
                    payload['circles'].append(
                        {'c': uv.tolist(), 'r': 11, 'rgb': self.goal_rgb, 'w': 2})
                    payload['circles'].append(
                        {'c': uv.tolist(), 'r': 3, 'rgb': self.goal_rgb, 'w': -1})
                    payload['labels'].append(
                        {'p': [int(uv[0]) + 14, int(uv[1]) - 8], 't': 'GOAL',
                         'rgb': self.goal_rgb, 's': 0.55})

        if lpath is not None:
            lpts = path_to_xyz(lpath)
            lframe = lpath.header.frame_id or frame_id
            Ml = M if lframe == frame_id else self._lookup(lframe)
            if Ml is not None:
                payload['polylines'].extend(
                    self._polylines_for(Ml, lpts, self.local_rgb, 2))

        # Projected fine, but every point failed the cull -- the plan is behind
        # the robot or off to the side. Without this the operator cannot tell
        # it apart from a plan that was never received.
        if not payload['polygons'] and not payload['polylines']:
            payload['hud'].append('plan outside camera view')

        self._send(payload)

    def _send(self, payload):
        try:
            self.sock.send(msgpack.packb(payload, use_bin_type=True),
                           flags=zmq.NOBLOCK)
            self.sent += 1
        except zmq.Again:
            # Renderer down or the link is congested. Dropping is correct.
            self.dropped += 1
        except Exception as exc:
            self.dropped += 1
            self.get_logger().warn(f'overlay send failed: {exc}',
                                   throttle_duration_sec=5.0)

    def _report(self):
        if self.sent == 0 and self.dropped > 0:
            self.get_logger().warn(
                f'{self.dropped} overlay frames dropped and none sent -- is '
                f'jetson_path_overlay.py running and bound on {self.overlay_url}?')
        self.sent = 0
        self.dropped = 0

    def destroy_node(self):
        try:
            self.sock.close()
            self.zmq_context.term()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PathOverlayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
