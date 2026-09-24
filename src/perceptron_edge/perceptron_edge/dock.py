#!/usr/bin/env python3
"""dock - drive onto the charging dock and prove it is charging.

    robot dock X Y YAW   Nav2 to the pre-dock pose (map frame), then the camera
                         brings the robot's nose onto the contacts
    robot dock           the same, with the pre-dock pose used last time
    robot dock here      no Nav2: the robot is already in front of the dock
    robot undock [M]     reverse straight off the dock (default 0.35 m)
    robot dock-teach     robot ON the dock and charging: record the board's pose
    robot dock-watch     live board pose, errors and current; never moves
    robot goal X Y YAW   (also here) Nav2 through the action, with its outcome
    robot pose X Y YAW   (also here) seed AMCL once AMCL is listening

WHY NAV2 ONLY GETS IT CLOSE

The dock stands against a wall, inside the costmap inflation, and the contacts
need +-3 cm. So Nav2 drives to the pre-dock pose in front of the dock and the
2x2 ArUco board on the dock does the rest.

1. The docked pose is TAUGHT, not modelled. `dock-teach`, run while the robot
   sits on the charger and charging, records where the robot is relative to
   the board. Docking reproduces that relative pose. The reference came through
   the same camera and the same maths, so errors in the camera mount, the
   board's lean or the intrinsics cancel at the target; they only bend the path.

   Teach also calibrates the camera: it reverses 25 cm in a straight line and
   finds the rotation that makes the board slide straight back. The URDF mount
   was 5.2 deg off in tilt, which made views from 0.7 m useless until fixed.

2. Each view is solved for 3 unknowns only - the robot's x, y, yaw relative to
   the docked pose - with the board's 3D pose from teach held fixed. A free PnP
   of a small board seen head-on has a mirror solution (measured: -5.3 deg for
   a robot square to the board); this one does not. Each view is placed in odom
   with the odometry from the moment the frame was taken, and the docked pose
   is a range-weighted median of the last 6 s of views. The dock does not move,
   so that costs no lag; the 20 Hz control loop steers against it with fresh
   50 Hz odometry, so camera latency never enters the loop.

3. Line up first, then drive in. A differential drive cannot move sideways, so
   a sideways error is removed by turn-drive-turn onto the dock axis 0.22 m
   out (views beyond ~0.6 m are poor). The run-in follows the axis slowly,
   steering with

       w = -v (e_y / L^2 + 2 e_yaw / L)

   which is critically damped over a distance L at any speed v, and stops once
   10 cm out to look again from close range before touching.

4. Charging is the only proof of docking. No charge after contact: push 1 cm
   more, then wiggle, then back off and line up again aiming slightly left or
   right. Up to `max_attempts` times. Odometry dropouts (the STM32 USB link
   wedging under motor load) stop the robot and are waited out.

Talks to the serial bridge directly over ZMQ for the camera (ArUco corners,
switched on only while this runs) and the battery (100 Hz, so contact is seen
within ~0.15 s), and to ROS for /odom, /scan, /cmd_vel and Nav2.
"""

import argparse
import math
import os
import signal
import statistics
import sys
import threading
import time
from collections import deque, namedtuple

import cv2
import msgpack
import numpy as np
import yaml
import zmq

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from perceptron_navigation import marker_map as mm

WS = os.environ.get('PERCEPTRON_WS', '/opt/perceptron_ws')
STATION_PATH = os.path.join(WS, 'config', 'dock_station.yaml')
DT = 0.05   # control period, s


# One clean view of the board: wall time, image time, pose (robot in the board
# frame for free solves, relative to the docked pose for docked solves), tiles
# used, reprojection px, range m, and base_from_board (free solves only).
Seen = namedtuple('Seen', 't t_img pose tiles err rng base_from_board')


class DockError(Exception):
    """Docking cannot continue; the message says why, for a human."""


class Cancelled(Exception):
    pass


# ------------------------------------------------------------------ SE(2)

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def compose(a, b):
    """Pose b, given in frame a, expressed in a's parent frame."""
    x, y, t = a
    c, s = math.cos(t), math.sin(t)
    return (x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], wrap(t + b[2]))


def inverse(a):
    x, y, t = a
    c, s = math.cos(t), math.sin(t)
    return (-c * x - s * y, s * x - c * y, -t)


def relative(a, b):
    """Pose b expressed in frame a (both in the same parent)."""
    return compose(inverse(a), b)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def quaternion_rotation(q):
    return quaternion_matrix(q.x, q.y, q.z, q.w)


def quaternion_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_quaternion(r):
    """(x, y, z, w) of a 3x3 rotation."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0, r[1, 0] - r[0, 1])
    return x, y, z, w


def mean_transform(transforms):
    """Average of nearly equal 4x4 rigid transforms (quaternion mean)."""
    quats = [np.array(matrix_quaternion(t[:3, :3])) for t in transforms]
    ref = quats[0]
    q = sum(v if v.dot(ref) >= 0 else -v for v in quats)
    q = q / np.linalg.norm(q)
    return mm.make_transform(quaternion_matrix(*q), np.mean([t[:3, 3] for t in transforms], axis=0))


def fit_camera_correction(track):
    """Rotation that makes the board slide straight back while the robot reverses.

    track: [(robot forward displacement m, board centre in base via the URDF
    camera mount)]. A straight reverse moves the board by exactly -d along the
    robot's x axis, so the fitted direction of travel must be (-1, 0, 0); the
    smallest rotation taking it there is the mount's real pitch and pan error.
    Roll is not observable this way and is left alone.
    """
    if len(track) < 25:
        return None, 'only %d usable views' % len(track)
    d = np.array([t[0] for t in track])
    points = np.array([t[1] for t in track])
    span = float(d.max() - d.min())
    if span < 0.12:
        return None, 'the board was only tracked over %.0f cm' % (span * 100)
    design = np.vstack([d, np.ones(len(d))]).T
    coef = np.linalg.lstsq(design, points, rcond=None)[0]
    slope = coef[0]
    rms = float(np.sqrt(np.mean(np.sum((points - design.dot(coef)) ** 2, axis=1))))
    scale = float(np.linalg.norm(slope))
    u = slope / scale
    target = np.array([-1.0, 0.0, 0.0])
    axis = np.cross(u, target)
    angle = math.atan2(float(np.linalg.norm(axis)), float(u.dot(target)))
    if angle > math.radians(15.0):
        return None, 'implausible %.0f deg correction - did the robot turn?' % math.degrees(angle)
    if angle < 1e-6:
        rotation = np.eye(3)
    else:
        rotation, _ = cv2.Rodrigues((axis / np.linalg.norm(axis) * angle).reshape(3, 1))
    return rotation, {'frames': len(track), 'span_m': span, 'rms_mm': rms * 1e3, 'scale': scale}


def weighted_median(values, weights):
    order = sorted(range(len(values)), key=lambda i: values[i])
    half = sum(weights) / 2.0
    total = 0.0
    for i in order:
        total += weights[i]
        if total >= half:
            return values[i]
    return values[order[-1]]


def circular_mean(angles):
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))


# ------------------------------------------------------------ serial bridge

class BridgeFeed(threading.Thread):
    """ArUco corners, battery and heartbeat straight from jetson_robot_bridge.py.

    Direct rather than through ROS: the ArUco corners never become a ROS topic,
    and the battery arrives at 100 Hz here but is thinned to 2 Hz for ROS, which
    would be 2 cm of extra push at run-in speed before contact is noticed.
    """

    def __init__(self, cfg):
        super().__init__(daemon=True, name='bridge_feed')
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        for topic in (b'aruco', b'battery', b'heartbeat'):
            self.sub.setsockopt(zmq.SUBSCRIBE, topic)
        self.sub.setsockopt(zmq.RCVTIMEO, 300)
        self.sub.connect('tcp://%s:%d' % (cfg['zmq_host'], cfg['telemetry_port']))
        self.cmd = self.ctx.socket(zmq.PUSH)
        self.cmd.setsockopt(zmq.LINGER, 300)
        self.cmd.setsockopt(zmq.SNDHWM, 10)
        self.cmd.connect('tcp://%s:%d' % (cfg['zmq_host'], cfg['cmd_port']))
        self.lock = threading.Lock()
        self.aruco = None
        self.aruco_count = 0
        self.battery = deque(maxlen=500)    # (t, volts, amps)
        self.heartbeat = None
        self.running = True

    def run(self):
        while self.running:
            try:
                frames = self.sub.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            if len(frames) < 2:
                continue
            topic = frames[0]
            try:
                data = msgpack.unpackb(frames[1], raw=False)
            except Exception:
                continue
            with self.lock:
                if topic == b'aruco':        # the prefix also matches b'aruco_debug'
                    self.aruco = data
                    self.aruco_count += 1
                elif topic == b'battery':
                    self.battery.append((float(data.get('stamp', time.time())),
                                         float(data.get('voltage', 0.0)),
                                         float(data.get('current', 0.0))))
                elif topic == b'heartbeat':
                    self.heartbeat = data

    def latest_aruco(self):
        with self.lock:
            return self.aruco_count, self.aruco

    def battery_median(self, window):
        """(volts, amps, samples) over the last `window` seconds."""
        cutoff = time.time() - window
        with self.lock:
            recent = [b for b in self.battery if b[0] >= cutoff]
        if not recent:
            return None, None, 0
        return (statistics.median(b[1] for b in recent),
                statistics.median(b[2] for b in recent), len(recent))

    def camera_status(self):
        """The bridge's ArUco thread status dict, or a string saying why there is none."""
        with self.lock:
            hb = self.heartbeat
        if hb is None:
            return 'no heartbeat'
        return hb.get('aruco') or 'no camera thread'

    def camera(self, on, hold=4.0, rate=10.0):
        try:
            self.cmd.send(msgpack.packb({'aruco_enable': bool(on), 'hold': float(hold),
                                         'rate': float(rate)}, use_bin_type=True),
                          flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass

    def close(self):
        self.running = False
        self.join(timeout=1.0)
        self.sub.close(0)
        self.cmd.close()
        self.ctx.term()


# ------------------------------------------------------------------ vision

class BoardTracker:
    """Where the docked pose is, in odom, from every detection of the dock board.

    Two solvers:

    free    6-DOF PnP, the board's pose relative to the camera. Used by teach,
            with the robot on the charger 0.36 m from the board, where it is
            steady to 0.03 deg.
    docked  3-DOF: the robot's (x, y, yaw) relative to the docked pose, with the
            board's full 3D pose relative to the docked robot taken from teach.
            A 2x2 board seen nearly head-on from 0.7 m has a mirror PnP solution
            (measured: the free solve settled on -5.3 deg for a robot that was
            square to within a degree). The mirror needs the board to lean the
            other way; with the lean fixed by teach and the robot on a flat
            floor it no longer fits, so this solve cannot flip. It also answers
            the question docking actually asks, relative to the docked pose.
    """

    def __init__(self, cfg, base_from_camera, base0_from_board=None, ref_range=0.36):
        self.cfg = cfg
        self.board = {'tile_size': float(cfg['tile_size']),
                      'tile_spacing': float(cfg['tile_spacing']),
                      'ids': {int(k): v for k, v in cfg['board_ids'].items()}}
        # The board's own planar frame: origin under its centre, +x out of the face.
        self.frame_from_board = mm.make_transform(mm.wall_board_rotation(0.0), (0.0, 0.0, 0.0))
        self.base_from_camera = base_from_camera
        self.cam_from_base = mm.invert_transform(base_from_camera)
        self.base0_from_board = base0_from_board   # taught; None: free solves only
        self.ref_range = ref_range
        self.camera_matrix = np.array(cfg['camera_matrix'], dtype=np.float64).reshape(3, 3)
        self.dist = np.array(cfg['dist_coeffs'], dtype=np.float64)
        self.cal_size = tuple(int(v) for v in cfg['calibration_size'])
        self.estimate = None      # the DOCKED robot pose in odom (x, y, yaw)
        self.views = deque()      # (wall time, docked pose seen in odom, weight)
        self.fused = 0
        self.last_fused = 0.0     # wall time
        self.last = None          # Seen
        self.measured = 0         # clean measurements, either solver
        self.reject = ''

    # ------------------------------------------------------------ solving

    def _correspondences(self, payload):
        cfg = self.cfg
        if payload.get('dict') != cfg['dictionary']:
            self.reject = 'camera sends %s, dock board is %s' % (payload.get('dict'), cfg['dictionary'])
            return None
        pairs = [(i, c) for i, c in zip(payload.get('ids') or [], payload.get('corners') or [])
                 if i in self.board['ids']]
        ids = [i for i, _ in pairs]
        if len(set(ids)) != len(ids):   # two boards with the same ids in view
            pairs = [(i, c) for i, c in pairs if ids.count(i) == 1]
        if len(pairs) < int(cfg['min_tiles']):
            self.reject = '%d of the board\'s tiles in view' % len(pairs)
            return None
        obj, img, used = mm.assemble_correspondences(
            self.board, [i for i, _ in pairs], [c for _, c in pairs])
        width = int(payload.get('width', self.cal_size[0]))
        height = int(payload.get('height', self.cal_size[1]))
        if not mm.corners_are_central(img, width, height, float(cfg['edge_margin'])):
            self.reject = 'board at the image edge'
            return None
        matrix = mm.scale_intrinsics(self.camera_matrix, self.cal_size, (width, height))
        return obj, img, used, matrix

    def _free_solutions(self, obj, img, matrix):
        """Both planar-PnP solutions as [(reproj px, cam_from_board)], best first."""
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), matrix, self.dist,
                flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return []
        out = []
        for k in range(n):
            if float(tvecs[k][2]) <= 0.0:
                continue
            rotation, _ = cv2.Rodrigues(rvecs[k])
            out.append((float(np.ravel(errs)[k]), mm.make_transform(rotation, tvecs[k].reshape(3))))
        return sorted(out, key=lambda e: e[0])

    def measure_free(self, payload):
        """(robot pose in the board frame, tiles, reproj px, range m, base_from_board) or None."""
        c = self._correspondences(payload)
        if c is None:
            return None
        obj, img, used, matrix = c
        solutions = self._free_solutions(obj, img, matrix)
        if not solutions:
            self.reject = 'PnP failed'
            return None
        err, cam_from_board = solutions[0]
        if err > float(self.cfg['max_reprojection_error']):
            self.reject = 'reprojection error %.1f px' % err
            return None
        rng = float(np.linalg.norm(cam_from_board[:3, 3]))
        if rng > float(self.cfg['max_range']):
            self.reject = 'board %.2f m away' % rng
            return None
        base_from_board = self.base_from_camera.dot(cam_from_board)
        rel = mm.robot_pose_in_map(self.frame_from_board, base_from_board)
        return rel, len(used), err, rng, base_from_board

    def _project(self, pose, obj_h):
        """Board corners, normalised camera coordinates, for the robot at `pose`
        relative to the docked pose. None if any is behind the camera."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        base_from_base0 = np.array([[c, s, 0.0, -(c * pose[0] + s * pose[1])],
                                    [-s, c, 0.0, s * pose[0] - c * pose[1]],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])
        pts = self.cam_from_base.dot(base_from_base0).dot(self.base0_from_board).dot(obj_h.T)
        if np.any(pts[2] < 0.05):
            return None
        return (pts[:2] / pts[2]).T

    def _refine(self, guess, obj_h, measured, scale):
        """Gauss-Newton on (x, y, yaw). Returns (pose, rms px) or None."""
        pose = np.array(guess, dtype=np.float64)
        for _ in range(12):
            proj = self._project(pose, obj_h)
            if proj is None:
                return None
            r = ((proj - measured) * scale).ravel()
            jac = np.empty((r.size, 3))
            for k in range(3):
                step = np.zeros(3)
                step[k] = 1e-5
                p = self._project(pose + step, obj_h)
                if p is None:
                    return None
                jac[:, k] = (((p - measured) * scale).ravel() - r) / 1e-5
            delta = np.linalg.solve(jac.T.dot(jac) + 1e-9 * np.eye(3), -jac.T.dot(r))
            delta[:2] = np.clip(delta[:2], -0.2, 0.2)
            delta[2] = clamp(delta[2], -0.3, 0.3)
            pose += delta
            pose[2] = wrap(pose[2])
            if np.abs(delta).max() < 1e-7:
                break
        proj = self._project(pose, obj_h)
        if proj is None:
            return None
        rms = float(np.sqrt(np.mean(np.sum(((proj - measured) * scale) ** 2, axis=1))))
        return (float(pose[0]), float(pose[1]), float(pose[2])), rms

    def measure_docked(self, payload, guess=None):
        """(robot pose relative to the docked pose, tiles, reproj px, range m, None) or None."""
        c = self._correspondences(payload)
        if c is None:
            return None
        obj, img, used, matrix = c
        measured = cv2.undistortPoints(img.reshape(-1, 1, 2), matrix, self.dist).reshape(-1, 2)
        obj_h = np.hstack([obj, np.ones((len(obj), 1))])
        starts = [guess] if guess is not None else []
        for _, cam_from_board in self._free_solutions(obj, img, matrix):
            base0_from_base = self.base0_from_board.dot(
                mm.invert_transform(self.base_from_camera.dot(cam_from_board)))
            starts.append((base0_from_base[0, 3], base0_from_base[1, 3],
                           math.atan2(base0_from_base[1, 0], base0_from_base[0, 0])))
        best = None
        for start in starts:
            solved = self._refine(start, obj_h, measured, matrix[0, 0])
            if solved is not None and (best is None or solved[1] < best[1]):
                best = solved
        if best is None:
            self.reject = 'PnP failed'
            return None
        pose, rms = best
        if rms > float(self.cfg['max_reprojection_error']):
            self.reject = 'reprojection error %.1f px' % rms
            return None
        board_in_cam = self._project(pose, np.array([[0.0, 0.0, 0.0, 1.0]]))
        rng = math.hypot(pose[0] - self.base0_from_board[0, 3], pose[1] - self.base0_from_board[1, 3])
        if board_in_cam is None or rng > float(self.cfg['max_range']):
            self.reject = 'board %.2f m away' % rng
            return None
        return pose, len(used), rms, rng, None

    # ------------------------------------------------------------ fusing

    def fuse(self, pose, odom_pose, rng):
        """Add one docked-relative view; re-estimate the docked pose in odom.

        The estimate is a weighted median over the last `estimate_window`
        seconds, weight (ref_range / range)^4: PnP heading error grows roughly
        with range squared, so as the robot closes in the near views take over
        within a frame or two, and a median cannot be dragged by a stray view
        the way a running average can (measured: one bad view after a reverse
        swung a blended estimate by 15 cm).
        """
        now = time.time()
        self.views.append((now, compose(odom_pose, inverse(pose)),
                           (self.ref_range / max(rng, 0.05)) ** float(self.cfg['range_weight_power'])))
        window = float(self.cfg['estimate_window'])
        while self.views and now - self.views[0][0] > window:
            self.views.popleft()
        ref = self.views[-1][1]
        weights = [v[2] for v in self.views]
        x = weighted_median([v[1][0] for v in self.views], weights)
        y = weighted_median([v[1][1] for v in self.views], weights)
        yaw = ref[2] + weighted_median([wrap(v[1][2] - ref[2]) for v in self.views], weights)
        self.estimate = (x, y, wrap(yaw))
        self.fused += 1
        self.last_fused = now
        return True


# --------------------------------------------------------------------- ROS

class DockNode(Node):

    def __init__(self, cfg):
        super().__init__('dock')
        self.lock = threading.Lock()
        self.odom = deque(maxlen=250)   # (t, x, y, yaw, v, w), 5 s at 50 Hz
        self.scan = None
        self.cmd_pub = self.create_publisher(Twist, cfg['cmd_vel_topic'], 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.status_pub = self.create_publisher(String, '/dock/status', latched)
        self.create_subscription(Odometry, cfg['odom_topic'], self._odom_cb, 20)
        self.create_subscription(LaserScan, cfg['scan_topic'], self._scan_cb, qos_profile_sensor_data)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav_feedback = None

    def _odom_cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.odom.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw,
                              msg.twist.twist.linear.x, msg.twist.twist.angular.z))

    def _scan_cb(self, msg):
        self.scan = (time.time(), msg)

    def latest_odom(self):
        with self.lock:
            return self.odom[-1] if self.odom else None

    def odom_at(self, t):
        """Odometry (x, y, yaw, w) interpolated at time t, or None."""
        with self.lock:
            hist = list(self.odom)
        if len(hist) < 2 or t < hist[0][0]:
            return None
        if t >= hist[-1][0]:
            last = hist[-1]
            return (last[1], last[2], last[3], last[5]) if t - last[0] < 0.3 else None
        for i in range(len(hist) - 1, 0, -1):
            a, b = hist[i - 1], hist[i]
            if a[0] <= t:
                f = (t - a[0]) / max(1e-6, b[0] - a[0])
                return (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]),
                        wrap(a[3] + f * wrap(b[3] - a[3])), a[5] + f * (b[5] - a[5]))
        return None


# ------------------------------------------------------------------ docking

class Docker:

    def __init__(self, cfg):
        self.cfg = cfg
        self.node = DockNode(cfg)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.feed = BridgeFeed(cfg)
        self.feed.start()
        self.abort = False
        self.deadline = None
        self.camera_wanted = False
        self.camera_sent = 0.0
        self.aruco_seen = 0
        self.nav_handle = None
        self.tracker = None
        self.station = None
        self.bias = 0.0

    # -------------------------------------------------------------- plumbing

    def say(self, text):
        print('[%s] %s' % (time.strftime('%H:%M:%S'), text), flush=True)
        self.node.status_pub.publish(String(data=text))

    def check(self):
        if self.abort:
            raise Cancelled()
        if self.deadline is not None and time.time() > self.deadline:
            raise DockError('gave up after %.0f s (total_timeout)' % self.cfg['total_timeout'])

    def tick(self):
        time.sleep(DT)
        self.check()
        if self.camera_wanted and time.time() - self.camera_sent > 1.0:
            self.feed.camera(True, hold=4.0, rate=self.cfg['camera_rate'])
            self.camera_sent = time.time()
        if self.tracker is not None:
            self._vision()

    def hold(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.tick()

    def send(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.node.cmd_pub.publish(msg)

    def stop(self):
        for _ in range(3):
            self.send(0.0, 0.0)
            time.sleep(0.02)

    def pose(self):
        o = self.node.latest_odom()
        if o is not None and time.time() - o[0] < 0.5:
            return o[1], o[2], o[3]
        # Odometry paused. On this robot that is nearly always the STM32 USB link
        # wedging under motor load; the bridge reopens it within ~5 s. Wait for
        # it rather than abandon the dock.
        self.send(0.0, 0.0)
        self.say('  odometry paused (STM32 serial link?) - stopped, waiting for it')
        end = time.time() + 12.0
        while time.time() < end:
            self.tick()
            o = self.node.latest_odom()
            if o is not None and time.time() - o[0] < 0.3:
                self.say('  odometry back')
                return o[1], o[2], o[3]
        raise DockError('no /odom for 12 s - the serial bridge lost the STM32 (robot logs)')

    def wait_for_odom(self):
        end = time.time() + 5.0
        while self.node.latest_odom() is None:
            if time.time() > end:
                raise DockError('no /odom - is `robot start` running?')
            time.sleep(0.1)
            self.check()

    def battery(self, window):
        """Median (volts, amps) over the last `window` s; waits for the feed to start."""
        end = time.time() + 3.0
        while True:
            volts, amps, n = self.feed.battery_median(window)
            if n >= 3:
                return volts, amps
            if time.time() > end:
                raise DockError('no battery readings from the serial bridge')
            time.sleep(0.1)
            self.check()

    def charging_now(self):
        _, amps, n = self.feed.battery_median(0.15)
        return n >= 3 and amps < self.cfg['charge_current']

    # ---------------------------------------------------------------- camera

    def setup_vision(self):
        cfg = self.cfg
        try:
            tf = self.node.tf_buffer.lookup_transform(
                cfg['base_frame'], cfg['camera_frame'], Time(), timeout=Duration(seconds=5.0))
        except Exception as exc:
            raise DockError('no TF %s -> %s (robot_state_publisher running?): %s'
                            % (cfg['base_frame'], cfg['camera_frame'], exc))
        t = tf.transform.translation
        rotation = quaternion_rotation(tf.transform.rotation)
        base0_from_board, ref_range = None, 0.36
        if self.station is not None:
            base0_from_board = self.station['base0_from_board']
            ref_range = float((self.station.get('teach_stats') or {}).get('range_m', 0.36))
            rotation = self.station['camera_correction'].dot(rotation)
        self.tracker = BoardTracker(cfg, mm.make_transform(rotation, (t.x, t.y, t.z)),
                                    base0_from_board, ref_range)

    def camera_on(self):
        self.camera_wanted = True
        self.camera_sent = 0.0

    def camera_off(self):
        self.camera_wanted = False
        self.feed.camera(False)

    def _vision(self):
        count, payload = self.feed.latest_aruco()
        if payload is None or count == self.aruco_seen:
            return
        self.aruco_seen = count
        if not payload.get('ids'):
            self.tracker.reject = 'no markers in view'
            return
        t_img = float(payload.get('stamp', time.time())) - self.cfg['camera_latency']
        odom = self.node.odom_at(t_img)
        if self.tracker.base0_from_board is None:      # teach, or watch before teach
            m = self.tracker.measure_free(payload)
            if m is not None:
                self.tracker.last = Seen(time.time(), t_img, *m)
                self.tracker.measured += 1
            return
        guess = None
        if self.tracker.estimate is not None and odom is not None:
            guess = relative(self.tracker.estimate, odom[:3])
        m = self.tracker.measure_docked(payload, guess)
        if m is None:
            return
        self.tracker.last = Seen(time.time(), t_img, *m)
        self.tracker.measured += 1
        if odom is None or abs(odom[3]) > self.cfg['max_turn_rate_for_vision']:
            return
        self.tracker.fuse(m[0], odom[:3], m[3])

    def wait_fresh(self, count, timeout):
        """Stand still until `count` new detections are fused. True if they were."""
        start = self.tracker.fused
        end = time.time() + timeout
        while time.time() < end:
            self.tick()
            if self.tracker.estimate is not None and self.tracker.fused - start >= count:
                return True
        return False

    def tracking(self):
        tr = self.tracker
        if tr.base0_from_board is None:      # free solves (teach): recent and repeated
            return tr.measured >= 3 and tr.last is not None and time.time() - tr.last.t < 0.5
        return tr.estimate is not None and tr.fused >= 3

    def acquire(self, timeout=8.0, search=True):
        """Camera on and the board found; True if it is being tracked."""
        self.camera_on()
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.tick()
            status = self.feed.camera_status()
            if status == 'no heartbeat' and time.time() - t0 > 3.0:
                raise DockError('the serial bridge is not publishing (no heartbeat)')
            if status == 'no camera thread':
                raise DockError('the serial bridge has no camera thread (started with '
                                '--no-aruco). `robot dock` restarts it correctly.')
            if self.tracking():
                return True
        if not search:
            return False
        self.say('board not in view - looking left and right')
        _, _, yaw0 = self.pose()
        for offset in (0.45, -0.45, 0.0):
            self.rotate_to(lambda: yaw0 + offset, tol=0.05, timeout=8.0)
            if self.wait_fresh(3, 1.5):
                return True
        return self.tracking()

    # ------------------------------------------------------------ geometry

    def goal(self):
        """Docked pose in odom (plus this attempt's sideways aim offset)."""
        return compose(self.tracker.estimate, (0.0, self.bias, 0.0))

    def errors(self):
        """(along, sideways, heading) of the robot relative to the docked pose."""
        return relative(self.goal(), self.pose())

    def describe(self, e):
        return ('%.1f cm to go, %+.1f cm sideways, %+.1f deg'
                % (-e[0] * 100.0, e[1] * 100.0, math.degrees(e[2])))

    # ------------------------------------------------------------- motions

    def rotate_to(self, target, tol=0.025, timeout=10.0):
        """Turn on the spot until yaw = target() (odom). False on timeout."""
        cfg = self.cfg
        t0 = time.time()
        settled = 0
        while True:
            self.tick()
            err = wrap(target() - self.pose()[2])
            if abs(err) < tol:
                self.send(0.0, 0.0)
                settled += 1
                if settled >= 4:
                    return True
                continue
            settled = 0
            if time.time() - t0 > timeout:
                self.stop()
                return False
            w = clamp(cfg['turn_gain'] * err, -cfg['w_max'], cfg['w_max'])
            if abs(w) < cfg['w_min']:
                w = math.copysign(cfg['w_min'], err)
            self.send(0.0, w)

    def drive_to(self, target, tol=0.01, timeout=20.0):
        """Drive to the odom point target() = (x, y), facing it."""
        cfg = self.cfg
        t0 = time.time()
        while True:
            self.tick()
            px, py = target()
            x, y, yaw = self.pose()
            dist = math.hypot(px - x, py - y)
            heading = wrap(math.atan2(py - y, px - x) - yaw)
            along = dist * math.cos(heading)
            if dist < tol:
                self.stop()
                return True
            if time.time() - t0 > timeout:
                self.stop()
                return False
            if abs(heading) > 0.35 and dist > 0.04:
                self.rotate_to(lambda: math.atan2(target()[1] - self.pose()[1],
                                                  target()[0] - self.pose()[0]))
                continue
            if along < 0.003:     # facing it and level with it (or just past)
                self.stop()
                return True
            v = clamp(1.2 * along, 0.03, cfg['v_stage'])
            w = 0.0 if dist < 0.04 else clamp(2.0 * heading, -cfg['w_max'], cfg['w_max'])
            self.send(v, w)

    def rear_blocked(self):
        scan = self.node.scan
        if scan is None or time.time() - scan[0] > 1.0:
            return 'no fresh /scan to check behind the robot'
        msg = scan[1]
        half = math.radians(self.cfg['rear_half_angle_deg'])
        for i, r in enumerate(msg.ranges):
            if not (msg.range_min < r < self.cfg['rear_clearance']):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if abs(wrap(angle - math.pi)) < half:
                return 'something %.2f m behind the robot' % r
        return None

    def reverse(self, distance, speed=None):
        """Back up `distance` m holding the current heading, watching behind."""
        cfg = self.cfg
        speed = speed or cfg['v_reverse']
        x0, y0, yaw0 = self.pose()
        t0 = time.time()
        while True:
            self.tick()
            x, y, yaw = self.pose()
            done = -((x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0))
            if done >= distance - 0.003:
                self.stop()
                return
            blocked = self.rear_blocked()
            if blocked:
                self.stop()
                raise DockError('stopped reversing: %s' % blocked)
            if time.time() - t0 > 4.0 + distance / 0.02:
                self.stop()
                raise DockError('reversing is not moving the robot (%.2f of %.2f m)' % (done, distance))
            v = -clamp(1.2 * (distance - done), 0.03, speed)
            self.send(v, clamp(2.0 * wrap(yaw0 - yaw), -0.3, 0.3))

    # ------------------------------------------------------------ the steps

    def line_up(self):
        """Put the robot on the dock axis, facing the dock, at least min_run_in out."""
        cfg = self.cfg
        stage = cfg['staging_distance']
        e = None
        for step in range(int(cfg['max_align_steps'])):
            if not self.wait_fresh(3, 3.0):
                self.say('  (board not in view - steering on odometry)')
            e = self.errors()
            self.say('  lining up: ' + self.describe(e))
            if e[0] > -cfg['min_run_in']:
                self.say('  too close to correct - backing off to %.0f cm' % (stage * 100))
                self.reverse(stage + e[0])
                continue
            aligned = abs(e[1]) <= cfg['lateral_tolerance']
            if aligned and abs(e[2]) <= cfg['yaw_tolerance']:
                return
            if aligned and e[0] > -(stage + 0.10):
                self.rotate_to(lambda: self.goal()[2])
                continue
            # Turn toward the axis point `stage` out, drive there, turn to face the dock.
            spot = lambda: compose(self.goal(), (-stage, 0.0, 0.0))[:2]  # noqa: E731
            x, y, _ = self.pose()
            if math.hypot(spot()[0] - x, spot()[1] - y) < 0.08:
                # A few cm, mostly sideways, cannot be driven to directly (it would be
                # a quarter turn, 2 cm, a quarter turn). Back off so it lies ahead at
                # a shallow angle instead.
                self.reverse(0.10)
            self.drive_to(spot)
            self.rotate_to(lambda: self.goal()[2])
        e = self.errors()
        if abs(e[1]) <= 1.5 * cfg['lateral_tolerance'] and abs(e[2]) <= 3 * cfg['yaw_tolerance']:
            return
        raise DockError('could not line up with the dock (%s)' % self.describe(e))

    def run_in(self):
        """Follow the dock axis in until charging, a stall or push_past. Returns why it stopped.

        Stops once at checkpoint_distance to look again from close range, where
        the board reads to a millimetre: too far off the axis -> 'offline' (the
        caller lines up again), crooked -> turn on the spot, then the last
        centimetres.
        """
        cfg = self.cfg
        length = cfg['steer_length']
        t0 = last_print = time.time()
        trail = deque()
        checked = False
        while True:
            self.tick()
            e = self.errors()
            now = time.time()
            if self.charging_now():
                self.stop()
                return 'charging'
            if e[0] >= cfg['push_past']:
                self.stop()
                return 'end'
            if not checked and e[0] > -cfg['checkpoint_distance']:
                checked = True
                self.stop()
                self.wait_fresh(6, 2.0)
                e = self.errors()
                self.say('  checkpoint: ' + self.describe(e))
                if abs(e[1]) > cfg['checkpoint_lateral']:
                    return 'offline'
                if abs(e[2]) > cfg['checkpoint_yaw']:
                    self.rotate_to(lambda: self.goal()[2], tol=0.012, timeout=4.0)
                trail.clear()
                continue
            x, y, _ = self.pose()
            trail.append((now, x, y))
            while trail and now - trail[0][0] > 1.0:
                trail.popleft()
            if (e[0] > -0.08 and now - trail[0][0] >= 0.9
                    and math.hypot(x - trail[0][1], y - trail[0][2]) < 0.004):
                self.stop()
                return 'stalled'
            if now - t0 > cfg['run_in_timeout']:
                self.stop()
                return 'timeout'
            if now - last_print > 1.0:
                self.say('  driving in: ' + self.describe(e))
                last_print = now
            v = clamp(0.8 * (cfg['push_past'] - e[0]), cfg['v_contact'], cfg['v_final'])
            if checked:
                # Last centimetres: square to the dock, no chasing a sub-cm offset.
                # Measured: steering out 0.7 cm over the last 8 cm turned the
                # robot 3 deg just as it met the contacts.
                w = -2.0 * e[2]
            else:
                w = -v * (e[1] / length ** 2 + 2.0 * e[2] / length)
            self.send(v, clamp(w, -cfg['w_final_max'], cfg['w_final_max']))

    def verify(self, baseline, quick=False):
        """Stopped: is the charger feeding the battery? (ok, description)"""
        cfg = self.cfg
        self.stop()
        window = 0.5 if quick else cfg['verify_time']
        self.hold((0.5 if quick else cfg['settle_time']) + window)
        volts, amps = self.battery(window)
        if amps < cfg['charge_current']:
            return True, 'charging at %.2f A (%.2f V)' % (amps, volts)
        if (baseline is not None and volts >= cfg['full_battery_voltage']
                and amps < baseline[1] - cfg['full_battery_drop']
                and volts > baseline[0] + cfg['full_battery_rise']):
            return True, ('charger connected: %.2f A, %.2f V (was %.2f A, %.2f V) - battery nearly full'
                          % (amps, volts, baseline[1], baseline[0]))
        return False, '%.2f A, %.2f V - not charging' % (amps, volts)

    def nudge_and_wiggle(self, baseline):
        """In contact but no charge: push a little further, then rock side to side."""
        cfg = self.cfg
        x0, y0, yaw0 = self.pose()
        t0 = time.time()
        while time.time() - t0 < 2.0:
            self.tick()
            x, y, _ = self.pose()
            if math.hypot(x - x0, y - y0) >= cfg['nudge_distance'] or self.charging_now():
                break
            self.send(cfg['v_contact'], 0.0)
        ok, why = self.verify(baseline, quick=True)
        if ok:
            ok, why = self.verify(baseline)     # a quick yes has to hold for the full check
            if ok:
                return ok, why + ' (after a nudge)'
        for offset in (cfg['wiggle_angle'], -cfg['wiggle_angle'], 0.0):
            self.rotate_to(lambda: yaw0 + offset, tol=0.012, timeout=2.5)
            ok, why = self.verify(baseline, quick=True)
            if ok:
                ok, why = self.verify(baseline)
                if ok:
                    return ok, why + ' (after a wiggle)'
        return False, why

    def dock_visual(self):
        cfg = self.cfg
        if not self.acquire(search=True):
            raise DockError('cannot see the dock board (%s). Is the robot in front of the dock?'
                            % (self.tracker.reject or 'nothing detected'))
        offsets = cfg['lateral_offsets'] or [0.0]
        for attempt in range(1, int(cfg['max_attempts']) + 1):
            self.bias = float(offsets[(attempt - 1) % len(offsets)])
            self.say('attempt %d/%d%s' % (attempt, cfg['max_attempts'],
                                          ', aiming %+.1f cm sideways' % (self.bias * 100)
                                          if self.bias else ''))
            self.line_up()
            self.stop()
            self.hold(1.0)      # motors wound down, or the baseline reads high and sagged
            baseline = self.battery(0.5)
            self.say('  lined up - driving in (%.2f A before contact)' % baseline[1])
            why_stopped = self.run_in()
            if why_stopped == 'offline':
                e = self.errors()
                self.say('  attempt %d: %.1f cm off the dock axis at the checkpoint - backing off '
                         'to line up again' % (attempt, e[1] * 100))
                self.reverse(max(0.05, cfg['staging_distance'] + e[0]))
                continue
            self.say('  stopped (%s): %s' % (why_stopped, self.describe(self.errors())))
            ok, why = self.verify(baseline)
            if not ok and why_stopped in ('end', 'stalled', 'charging'):
                self.say('  touching the dock but %s - nudging' % why)
                ok, why = self.nudge_and_wiggle(baseline)
            if ok:
                return attempt, why
            self.say('  attempt %d: %s (%s) - backing off to try again'
                     % (attempt, why, why_stopped))
            e = self.errors()
            self.reverse(max(0.05, cfg['staging_distance'] + e[0]))
        raise DockError('no charge after %d attempts. Check the contacts line up and the '
                        'charger is on, then `robot dock-teach` if the dock moved'
                        % cfg['max_attempts'])

    # -------------------------------------------------------------- Nav2

    def _nav_feedback(self, msg):
        self.node.nav_feedback = msg.feedback

    def navigate(self, x, y, yaw, what='the pre-dock pose', attempts=None):
        """NavigateToPose through the action server, not the /goal_pose topic.

        bt_navigator takes /goal_pose best-effort, and Foxglove subscribes to it
        too, so `ros2 topic pub -w 1` can publish the moment Foxglove matches,
        before Nav2 does: the goal vanishes without a log line. The action has
        an accept handshake, so a goal either starts or says why not.
        """
        cfg = self.cfg
        attempts = int(attempts or cfg['nav_attempts'])
        if not self.node.nav.wait_for_server(timeout_sec=5.0):
            raise DockError('Nav2 is not running (robot start nav map=NAME)')
        for attempt in range(1, attempts + 1):
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = 'map'   # stamp 0: latest map -> base_footprint
            goal.pose.pose.position.x = float(x)
            goal.pose.pose.position.y = float(y)
            goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
            goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
            self.say('Nav2: driving to %s x=%.3f y=%.3f yaw=%.2f%s'
                     % (what, x, y, yaw, '' if attempt == 1 else ' (try %d)' % attempt))
            future = self.node.nav.send_goal_async(goal, feedback_callback=self._nav_feedback)
            end = time.time() + 10.0
            while not future.done():
                if time.time() > end:
                    raise DockError('Nav2 did not answer the goal')
                time.sleep(0.05)
                self.check()
            handle = future.result()
            if handle is None or not handle.accepted:
                self.say('Nav2 refused the goal - not active yet? It activates once AMCL has a '
                         'pose (robot pose X Y YAW, or Publish pose estimate in Foxglove)')
                self.hold(2.0)
                continue
            self.say('Nav2: goal accepted')
            self.nav_handle = handle
            result = handle.get_result_async()
            last_print = time.time()
            while not result.done():
                time.sleep(0.1)
                self.check()
                fb = self.node.nav_feedback
                if fb is not None and time.time() - last_print > 3.0:
                    self.say('Nav2: %.2f m to go' % fb.distance_remaining)
                    last_print = time.time()
            self.nav_handle = None
            status = result.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.say('Nav2: arrived at %s' % what)
                return
            self.say('Nav2: goal %s' % {GoalStatus.STATUS_ABORTED: 'failed (no path, or stuck)',
                                         GoalStatus.STATUS_CANCELED: 'cancelled'}.get(status, status))
            if status == GoalStatus.STATUS_CANCELED:
                raise DockError('Nav2 goal was cancelled')
            if attempt < attempts:
                self.hold(2.0)
        raise DockError('Nav2 could not reach %s. Is it free space? Watch '
                        '/global_costmap/costmap in Foxglove; `robot logs` says why.' % what)

    # ------------------------------------------------------------ commands

    def load_station(self):
        if not os.path.exists(STATION_PATH):
            raise DockError('no %s yet. Put the robot on the dock (charging) and run: '
                            'robot dock-teach' % STATION_PATH)
        with open(STATION_PATH) as f:
            st = yaml.safe_load(f) or {}
        b = st.get('board_in_base') or {}
        try:
            rot = quaternion_matrix(*[float(b[k]) for k in ('qx', 'qy', 'qz', 'qw')])
            st['base0_from_board'] = mm.make_transform(rot, (float(b['x']), float(b['y']), float(b['z'])))
        except (KeyError, TypeError, ValueError):
            raise DockError('%s has no board_in_base - put the robot on the charger and run: '
                            'robot dock-teach' % STATION_PATH)
        c = st.get('camera_correction') or {}
        st['camera_correction'] = (quaternion_matrix(*[float(c[k]) for k in ('qx', 'qy', 'qz', 'qw')])
                                   if c else np.eye(3))
        self.station = st

    def start(self):
        self.wait_for_odom()
        self.deadline = time.time() + self.cfg['total_timeout']
        self.setup_vision()

    def near_dock(self):
        if self.tracker.estimate is None:
            return False
        e = self.errors()
        return (-self.cfg['near_dock_range'] < e[0] < 0.05 and abs(e[1]) < 0.3
                and abs(e[2]) < 0.8)

    def cmd_dock(self, predock):
        self.load_station()
        if predock is None:
            p = self.station.get('predock')
            if not p:
                raise DockError('no pre-dock pose yet: robot dock X Y YAW (map frame, in '
                                'front of the dock)')
            predock = (float(p['x']), float(p['y']), float(p['yaw']))
        elif predock != 'here':
            self.save_station(predock=predock)
        self.start()
        volts, amps = self.battery(0.5)
        if amps < self.cfg['charge_current']:
            self.say('DOCKED: already on the dock and charging (%.2f A, %.2f V)' % (amps, volts))
            return 0
        t0 = time.time()
        if predock != 'here':
            if self.acquire(timeout=6.0, search=False) and self.near_dock():
                self.say('the dock is %.2f m ahead - no need for Nav2' % -self.errors()[0])
            else:
                self.camera_off()   # ~0.4 of a core saved while Nav2 drives
                self.navigate(*predock)
        attempt, why = self.dock_visual()
        self.camera_off()
        self.say('DOCKED: %s - attempt %d, %.0f s' % (why, attempt, time.time() - t0))
        return 0

    def cmd_goal(self, x, y, yaw):
        """`robot goal`: drive there with Nav2 and report how it went."""
        self.navigate(x, y, yaw, what='the goal', attempts=1)
        return 0

    def cmd_pose(self, x, y, yaw):
        """`robot pose`: seed AMCL, publishing only once AMCL itself is listening."""
        # Transient-local: matches AMCL (volatile) and Foxglove, which subscribes
        # transient-local; a volatile publisher never matches Foxglove.
        pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        end = time.time() + 10.0
        while True:
            subs = self.node.get_subscriptions_info_by_topic('/initialpose')
            if any(s.node_name == 'amcl' for s in subs) and pub.get_subscription_count() >= len(subs):
                break
            if time.time() > end:
                raise DockError('AMCL is not listening on /initialpose - is `robot start nav` running?')
            time.sleep(0.1)
            self.check()
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.25 ** 2       # 25 cm
        cov[35] = 0.26 ** 2               # 15 deg
        msg.pose.covariance = cov
        for _ in range(2):
            msg.header.stamp = self.node.get_clock().now().to_msg()
            pub.publish(msg)
            time.sleep(0.3)
        self.say('pose sent to AMCL: x=%.3f y=%.3f yaw=%.2f - it refines that from the scan'
                 % (x, y, yaw))
        return 0

    def cmd_undock(self, distance):
        self.wait_for_odom()
        volts, amps = self.battery(0.5)
        self.say('reversing %.2f m off the dock (%.2f A now)' % (distance, amps))
        self.reverse(distance)
        self.hold(1.0)
        volts, amps = self.battery(0.8)
        self.say('undocked: %.2f A, %.2f V - Nav2 can plan from here' % (amps, volts))
        return 0

    def collect(self, count, timeout):
        """The next `count` new views (Seen), or as many as `timeout` allows."""
        samples = []
        seen = self.tracker.last.t if self.tracker.last else 0.0
        end = time.time() + timeout
        while len(samples) < count and time.time() < end:
            self.tick()
            last = self.tracker.last
            if last is not None and last.t != seen:
                seen = last.t
                samples.append(last)
        return samples

    def cmd_teach(self):
        """Record the docked view, calibrate the camera tilt, drive back on.

        1. On the charger: 40 views of the board -> its pose in the camera frame.
        2. Reverse straight for calib_distance, watching the board. It must slide
           straight back along the robot's x axis; any climb or sideways drift is
           the camera mount's real tilt and pan differing from the URDF. The fit
           is the rotation that makes it straight (measured 6 deg the first time).
        3. Drive straight back until the current turns negative again.
        """
        cfg = self.cfg
        self.wait_for_odom()
        volts, amps = self.battery(1.0)
        if amps >= cfg['charge_current']:
            raise DockError('not charging (%.2f A, %.2f V). Put the robot on the dock so it '
                            'charges, then teach.' % (amps, volts))
        self.setup_vision()      # no station: free solves through the URDF camera mount
        if not self.acquire(search=False):
            raise DockError('cannot see the dock board from here (%s)' % self.tracker.reject)
        docked = self.collect(40, 8.0)
        if len(docked) < 20:
            raise DockError('only %d clean views of the board in 8 s' % len(docked))
        xs = [v.pose[0] for v in docked]
        ys = [v.pose[1] for v in docked]
        yaws = [v.pose[2] for v in docked]
        yaw = circular_mean(yaws)
        spread = (statistics.pstdev(xs), statistics.pstdev(ys),
                  statistics.pstdev([wrap(a - yaw) for a in yaws]))
        if spread[0] > 0.004 or spread[1] > 0.004 or spread[2] > 0.01:
            raise DockError('board pose too unsteady to teach (std %.1f / %.1f mm, %.2f deg) - '
                            'is the robot moving?' % (spread[0] * 1e3, spread[1] * 1e3,
                                                      math.degrees(spread[2])))
        cam0_from_board = mean_transform([self.tracker.cam_from_base.dot(v.base_from_board)
                                          for v in docked])
        volts, amps = self.battery(1.0)
        map_pose = None
        try:
            tf = self.node.tf_buffer.lookup_transform('map', cfg['base_frame'], Time(),
                                                      timeout=Duration(seconds=1.0))
            q = tf.transform.rotation
            map_pose = (tf.transform.translation.x, tf.transform.translation.y,
                        math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        except Exception:
            pass

        # -- 2. camera tilt, from a straight reverse
        distance = float(cfg['calib_distance'])
        self.say('docked view recorded (%d frames, std %.1f/%.1f mm, %.2f deg). Reversing %.0f cm '
                 'slowly to calibrate the camera tilt' % (len(docked), spread[0] * 1e3, spread[1] * 1e3,
                                                         math.degrees(spread[2]), distance * 100))
        x0, y0, yaw0 = self.pose()
        track = [(0.0, v.base_from_board[:3, 3].copy()) for v in docked]
        seen = self.tracker.last.t if self.tracker.last else 0.0
        t0 = time.time()
        while True:
            self.tick()
            x, y, yaw = self.pose()
            done = -((x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0))
            last = self.tracker.last
            if last is not None and last.t != seen:
                seen = last.t
                o = self.node.odom_at(last.t_img)
                if (o is not None and last.err < cfg['calib_max_reproj']
                        and last.rng < cfg['calib_max_range']):
                    track.append(((o[0] - x0) * math.cos(yaw0) + (o[1] - y0) * math.sin(yaw0),
                                  last.base_from_board[:3, 3].copy()))
            if done >= distance:
                break
            blocked = self.rear_blocked()
            if blocked or time.time() - t0 > 4.0 + distance / 0.01:
                self.stop()
                raise DockError('calibration reverse stopped: %s' % (blocked or 'not moving'))
            self.send(-cfg['calib_speed'], clamp(2.0 * wrap(yaw0 - yaw), -0.3, 0.3))
        self.stop()
        correction, info = fit_camera_correction(track)
        if correction is None:
            self.say('camera tilt NOT calibrated (%s) - using the URDF mount' % info)
            correction, info = np.eye(3), {}
        else:
            urdf_axis = self.tracker.base_from_camera[:3, 2]          # optical +z in base
            real_axis = correction.dot(urdf_axis)
            tilt = math.degrees(math.asin(clamp(real_axis[2], -1, 1)) - math.asin(clamp(urdf_axis[2], -1, 1)))
            pan = math.degrees(wrap(math.atan2(real_axis[1], real_axis[0])
                                    - math.atan2(urdf_axis[1], urdf_axis[0])))
            info['tilt_deg'], info['pan_deg'] = tilt, pan
            self.say('camera: looks %.1f deg %s and %.1f deg %s than the URDF says (%d views over '
                     '%.0f cm, fit %.1f mm rms, vision/odometry scale %.3f)'
                     % (abs(tilt), 'lower' if tilt < 0 else 'higher', abs(pan),
                        'further right' if pan < 0 else 'further left', info['frames'],
                        info['span_m'] * 100, info['rms_mm'], info['scale']))
        base_from_camera = mm.make_transform(correction.dot(self.tracker.base_from_camera[:3, :3]),
                                             self.tracker.base_from_camera[:3, 3])
        board = base_from_camera.dot(cam0_from_board)
        rel = mm.robot_pose_in_map(self.tracker.frame_from_board, board)
        self.save_station(board=board, robot_in_board=rel, correction=(correction, info), stats={
            'frames': len(docked), 'std_x_mm': round(spread[0] * 1e3, 2),
            'std_y_mm': round(spread[1] * 1e3, 2),
            'std_yaw_deg': round(math.degrees(spread[2]), 3),
            'tiles': min(v.tiles for v in docked),
            'reproj_px': round(statistics.median(v.err for v in docked), 3),
            'range_m': round(statistics.mean(v.rng for v in docked), 3),
            'current_a': round(amps, 3), 'voltage_v': round(volts, 3)},
            dock_map_pose=map_pose)
        self.say('taught: docked, the robot sits %.3f m in front of the board, %+.3f m sideways, '
                 'heading %+.1f deg. Saved %s'
                 % (rel[0], rel[1], math.degrees(wrap(rel[2] - math.pi)), STATION_PATH))

        # -- 3. back onto the charger along the same line
        self.camera_off()
        self.say('driving back onto the charger')
        x0, y0, _ = self.pose()
        t0 = time.time()
        while not self.charging_now():
            self.tick()
            x, y, yaw = self.pose()
            if math.hypot(x - x0, y - y0) > distance + 0.04 or time.time() - t0 > 4.0 + distance / 0.01:
                self.stop()
                self.say('taught, but driving straight back did not reach the contacts - try: '
                         'robot dock here')
                return 1
            self.send(cfg['calib_speed'], clamp(2.0 * wrap(yaw0 - yaw), -0.3, 0.3))
        ok, why = self.verify(None)
        self.say(('back on the charger: ' if ok else 'back at the charger but ') + why)
        return 0 if ok else 1

    def cmd_watch(self):
        self.wait_for_odom()
        try:
            self.load_station()
        except DockError as exc:
            self.say('(%s) - showing the board pose only' % exc)
            self.station = None
        self.setup_vision()
        self.camera_on()
        self.say('watching the dock board (Ctrl-C to stop; the robot does not move)')
        last = 0.0
        while True:
            self.tick()
            if time.time() - last < 0.5:
                continue
            last = time.time()
            volts, amps, _ = self.feed.battery_median(0.5)
            power = ('%.2f A %.2f V %s' % (amps, volts, 'CHARGING' if amps < self.cfg['charge_current']
                                           else 'not charging')) if amps is not None else 'no battery data'
            t = self.tracker.last
            if t is None or time.time() - t.t > 1.0:
                cam = self.feed.camera_status()
                state = cam.get('reason') if isinstance(cam, dict) else cam
                print('board: not seen (%s; camera: %s) | %s'
                      % (self.tracker.reject or '-', state, power), flush=True)
                continue
            pose, tiles, err, rng = t.pose, t.tiles, t.err, t.rng
            if self.station is None:
                line = ('board %.3f m, %d tiles, %.2f px | robot in board frame x=%.3f y=%+.3f '
                        'heading %+.1f deg' % (rng, tiles, err, pose[0], pose[1],
                                               math.degrees(wrap(pose[2] - math.pi))))
            else:
                line = 'board %.2f m, %d tiles, %.2f px | this frame: %s' % (
                    rng, tiles, err, self.describe(pose))
                if self.tracker.estimate is not None:
                    line += ' | fused: ' + self.describe(self.errors())
            print(line + ' | ' + power, flush=True)

    def save_station(self, board=None, robot_in_board=None, stats=None, dock_map_pose=None,
                     predock=None, correction=None):
        st = {}
        if os.path.exists(STATION_PATH):
            with open(STATION_PATH) as f:
                st = yaml.safe_load(f) or {}
        if board is not None:
            q = matrix_quaternion(board[:3, :3])
            st['board_in_base'] = {'x': round(float(board[0, 3]), 5), 'y': round(float(board[1, 3]), 5),
                                   'z': round(float(board[2, 3]), 5),
                                   'qx': round(q[0], 6), 'qy': round(q[1], 6),
                                   'qz': round(q[2], 6), 'qw': round(q[3], 6)}
            st['robot_in_board'] = {'x': round(robot_in_board[0], 4), 'y': round(robot_in_board[1], 4),
                                    'yaw': round(robot_in_board[2], 5)}
            st['taught'] = time.strftime('%Y-%m-%d %H:%M:%S')
            st['teach_stats'] = stats
            rotation, info = correction if correction is not None else (np.eye(3), {})
            q = matrix_quaternion(rotation)
            st['camera_correction'] = dict({'qx': round(q[0], 6), 'qy': round(q[1], 6),
                                            'qz': round(q[2], 6), 'qw': round(q[3], 6)},
                                           **{k: round(float(v), 3) for k, v in info.items()})
            if dock_map_pose is not None:
                st['dock_map_pose'] = {'x': round(dock_map_pose[0], 3), 'y': round(dock_map_pose[1], 3),
                                       'yaw': round(dock_map_pose[2], 3)}
        if predock is not None:
            st['predock'] = {'x': float(predock[0]), 'y': float(predock[1]), 'yaw': float(predock[2])}
        os.makedirs(os.path.dirname(STATION_PATH), exist_ok=True)
        tmp = STATION_PATH + '.tmp'
        with open(tmp, 'w') as f:
            f.write('# Charging dock, written by `robot dock-teach` and `robot dock X Y YAW`.\n'
                    '# board_in_base: the dock board\'s 3D pose in base_footprint while docked\n'
                    '#   and charging - THE docking target. robot_in_board: the same, planar and\n'
                    '#   inverted (board frame: origin under the board centre, +x out of its face).\n'
                    '# camera_correction: rotation applied to the URDF camera mount, measured by\n'
                    '#   teach from a straight reverse (pitch/pan in degrees are for reading).\n'
                    '# predock: map-frame pose Nav2 drives to before the camera takes over.\n'
                    '# dock_map_pose: where AMCL put the robot when it was taught (info only).\n')
            yaml.safe_dump(st, f, default_flow_style=False, sort_keys=True)
        os.replace(tmp, STATION_PATH)

    def shutdown(self):
        try:
            self.stop()
        except Exception:
            pass
        if self.nav_handle is not None:
            try:
                self.nav_handle.cancel_goal_async()
                time.sleep(0.3)
            except Exception:
                pass
        if self.camera_wanted:
            self.camera_off()
        time.sleep(0.1)
        self.feed.close()
        self.executor.shutdown(timeout_sec=1.0)
        # Join before rclpy shuts down: a spin thread still inside the executor
        # at interpreter exit aborts the process ("terminate called without an
        # active exception") about half the time.
        self.spin_thread.join(timeout=2.0)
        self.node.destroy_node()


def load_config():
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory('perceptron_edge'), 'config', 'dock.yaml')
    with open(path) as f:
        return yaml.safe_load(f)['dock']


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='dock', description=__doc__.split('\n\n')[0])
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override a config/dock.yaml value for this run (YAML syntax)')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('go', help='Nav2 to the pre-dock pose, then dock on camera')
    p.add_argument('pose', nargs='*', type=float, metavar='X Y YAW')
    sub.add_parser('here', help='dock on camera from where the robot is')
    p = sub.add_parser('goal', help='Nav2 to X Y YAW (map frame) and report the outcome')
    p.add_argument('pose', nargs=3, type=float, metavar='X Y YAW')
    p = sub.add_parser('pose', help='seed AMCL with X Y YAW (map frame)')
    p.add_argument('pose', nargs=3, type=float, metavar='X Y YAW')
    p = sub.add_parser('undock', help='reverse off the dock')
    p.add_argument('distance', nargs='?', type=float)
    sub.add_parser('teach', help='on the dock and charging: record the docked pose')
    sub.add_parser('watch', help='live readout, never moves')
    args = parser.parse_args(argv)
    if args.cmd == 'go' and len(args.pose) not in (0, 3):
        parser.error('go takes X Y YAW (map frame, metres and radians) or nothing')

    cfg = load_config()
    for item in args.set:
        key, _, value = item.partition('=')
        if key not in cfg:
            parser.error('--set: no such setting %r in dock.yaml' % key)
        cfg[key] = yaml.safe_load(value)
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    docker = Docker(cfg)

    def on_signal(signum, frame):
        docker.abort = True
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    code = 1
    try:
        if args.cmd == 'go':
            code = docker.cmd_dock(tuple(args.pose) if args.pose else None)
        elif args.cmd == 'here':
            code = docker.cmd_dock('here')
        elif args.cmd == 'goal':
            code = docker.cmd_goal(*args.pose)
        elif args.cmd == 'pose':
            code = docker.cmd_pose(*args.pose)
        elif args.cmd == 'undock':
            code = docker.cmd_undock(args.distance or cfg['undock_distance'])
        elif args.cmd == 'teach':
            code = docker.cmd_teach()
        elif args.cmd == 'watch':
            code = docker.cmd_watch()
    except Cancelled:
        docker.say('cancelled - robot stopped')
        code = 130
    except DockError as exc:
        docker.say('FAILED: %s' % exc)
        code = 1
    finally:
        docker.shutdown()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
