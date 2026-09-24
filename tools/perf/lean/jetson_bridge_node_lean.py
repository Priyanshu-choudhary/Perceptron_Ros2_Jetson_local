#!/usr/bin/env python3
"""
jetson_bridge_node.py
ROS 2 Humble Client Node communicating with Jetson Nano over local Wi-Fi via ZeroMQ.
Translates binary ZMQ streams into native ROS 2 topics:
  - Publishes /scan (LaserScan) from LD19
  - Publishes /odom (Odometry) from STM32
  - Publishes /imu/data_raw (Imu) from STM32
  - Publishes /battery/measured_voltage and /battery/measured_current
  - Subscribes to /cmd_vel (Twist) and sends commands to Jetson
"""

import collections
import os
import math
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import Float32, String
from std_srvs.srv import SetBool
from builtin_interfaces.msg import Time as TimeMsg
import tf2_ros
import zmq
import msgpack

from perceptron_hardware.gyro_conditioner import GyroConditioner
from perceptron_hardware import sensor_guard

POSE_COV_DIAG = (0.05, 0.05, 1e6, 1e6, 1e6, 0.10)
TWIST_COV_DIAG = (0.01, 1e6, 1e6, 1e6, 1e6, 0.05)


def _diag6(diag):
    cov = [0.0] * 36
    for i, v in enumerate(diag):
        cov[i * 6 + i] = float(v)
    return cov


def _stamp_from_seconds(t):
    """Seconds on THIS machine's clock -> builtin_interfaces/Time."""
    if t < 0.0:
        t = 0.0
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1000000000:
        sec += 1
        nanosec -= 1000000000
    return TimeMsg(sec=sec, nanosec=nanosec)


class JetsonClock:
    """Translates the Jetson's time.time() stamps into this machine's clock.

    Every payload carries the instant the Jetson SAMPLED it. This node used to
    throw that away and stamp on arrival instead, which folds the whole
    transport delay into the measurement time - and over Wi-Fi that delay is
    not a constant. It idles near a millisecond and spikes to hundreds under
    congestion.

    A scan delayed 300 ms but labelled "now" is handed to slam_toolbox as
    though the room had rotated by whatever the robot turned through in the
    meantime. The matcher duly rotates map -> odom to make it fit and then,
    believing its own correction, redraws the walls in the new place: the
    sudden spin on the spot, followed by a second set of boundaries.

    The two machines keep independent clocks, so (t_recv - t_send) is
    offset + delay, with delay >= 0 and unknown. The MINIMUM of that difference
    over a window is therefore the cleanest estimate of the offset available:
    it is the sample that happened to suffer least delay. Everything above the
    minimum is transport delay, which yields a corrected stamp AND a
    per-message age for free, with no NTP and no clock discipline anywhere.

    Two properties worth preserving if this is ever edited:
      * age is >= 0 by construction (the current sample is inside the window),
        so a corrected stamp can never land in the future and trip TF
        extrapolation.
      * a laptop clock that jumps BACKWARDS - a WSL suspend/resume will do it -
        creates a new minimum immediately and is absorbed on the next message.
        Drift the other way is absorbed as the window slides.
    """

    def __init__(self, window_s=3.0):
        # THE WINDOW MUST BE SHORTER THAN max_sensor_age / clock_drift_rate.
        #
        # The offset is the minimum over the window, so it is up to window_s
        # old. If the two clocks drift apart at r, every message is reported
        # as r * window_s late even on a perfect network, and once that
        # exceeds max_sensor_age EVERY sensor payload is dropped and the robot
        # goes silently blind.
        #
        # That is not hypothetical: on 2026-09-16 WSL's clock was found to run
        # 2.2% slow (22 ms/s), which at the old window_s of 30 s manufactured
        # 660 ms of phantom age against a 500 ms limit. /scan and /odom stopped
        # publishing entirely while the bridge sat there looking healthy.
        # At 3 s the same drift costs 66 ms, comfortably inside the limit.
        #
        # Shortening it is close to free: the stream runs at ~70 Hz, so 3 s
        # still holds ~200 samples to pick a minimum from, and the estimator
        # only needs one lightly-delayed sample to be right.
        self._window_s = float(window_s)
        # (t_recv, diff) held with diff strictly increasing: a monotonic deque.
        self._mono = collections.deque()

    def update(self, t_send, t_recv):
        """Feed one message. Returns (corrected_stamp, age), both seconds."""
        diff = t_recv - t_send
        # Sliding-window minimum, O(1) amortised: anything already queued that
        # is no better than this sample can never be the window minimum again.
        while self._mono and self._mono[-1][1] >= diff:
            self._mono.pop()
        self._mono.append((t_recv, diff))
        cutoff = t_recv - self._window_s
        while len(self._mono) > 1 and self._mono[0][0] < cutoff:
            self._mono.popleft()
        offset = self._mono[0][1]
        return t_send + offset, diff - offset


def yaw_to_quaternion(yaw: float) -> tuple:
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class JetsonBridgeNode(Node):
    def __init__(self):
        super().__init__('jetson_bridge_node')

        # Declare parameters
        default_jetson_ip = os.environ.get('JETSON_IP', '192.168.1.7')
        self.declare_parameter('jetson', '')
        self.declare_parameter('jetson_ip', default_jetson_ip)
        self.declare_parameter('telemetry_port', 5555)
        self.declare_parameter('cmd_port', 5556)
        self.declare_parameter('laser_frame_id', 'laser_frame')
        self.declare_parameter('base_frame_id', 'base_footprint')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('auto_arm', True)
        self.declare_parameter('use_imu', True)
        # Drop sensor payloads that reached us later than this. A late scan is
        # worse than no scan: SLAM acts on it and rewrites the map. 0 disables.
        self.declare_parameter('max_sensor_age', 0.5)
        self.declare_parameter('clock_window_s', 3.0)
        # The same gyro conditioning stm32_bridge_node applies. Without this the
        # LAN-bridge path published a raw, biased, unscaled gyro with an all-zero
        # covariance - and since robot.launch.py defaults to use_jetson:=true,
        # that was the path the robot actually ran.
        GyroConditioner.declare_parameters(self)

        jetson_param = str(self.get_parameter('jetson').value or '').strip()
        jetson_ip_param = str(self.get_parameter('jetson_ip').value or '').strip()
        self.jetson_ip = jetson_param if jetson_param else (jetson_ip_param or default_jetson_ip)
        self.telemetry_port = self.get_parameter('telemetry_port').value
        self.cmd_port = self.get_parameter('cmd_port').value
        self.laser_frame_id = self.get_parameter('laser_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.imu_frame_id = self.get_parameter('imu_frame_id').value
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.auto_arm = self.get_parameter('auto_arm').value
        self.use_imu = bool(self.get_parameter('use_imu').value)
        self.max_sensor_age = float(self.get_parameter('max_sensor_age').value)
        self.jetson_clock = JetsonClock(
            float(self.get_parameter('clock_window_s').value))
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.last_odom_time = None
        self.gyro = GyroConditioner(self)
        # msgpack over ZeroMQ over Wi-Fi carries no application checksum, so
        # every float here is untrusted. A single NaN reaching the EKF poisons
        # it for the rest of the run - there is no recovery, only a restart.
        self._imu_rejects = sensor_guard.RejectCounter(self, 'imu')
        self._odom_rejects = sensor_guard.RejectCounter(self, 'odom')
        self._stale_rejects = sensor_guard.RejectCounter(self, 'stale telemetry')
        self.is_armed = False

        # ROS 2 Publishers
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 20)
        self.imu_pub = self.create_publisher(Imu, '/imu/data_raw', 20)
        self.volt_pub = self.create_publisher(Float32, '/battery/measured_voltage', 10)
        self.curr_pub = self.create_publisher(Float32, '/battery/measured_current', 10)
        self.status_pub = self.create_publisher(String, '/jetson/status', 10)

        # TF Broadcaster
        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Services for Arm/Disarm
        self.create_service(SetBool, '/arm_motors', self._arm_motors_cb)
        self.create_service(SetBool, '/robot/arm', self._arm_motors_cb)

        # ROS 2 Subscriber for /cmd_vel
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 10)

        # ZeroMQ Setup
        self.zmq_context = zmq.Context()

        # Telemetry SUB socket
        self.sub_socket = self.zmq_context.socket(zmq.SUB)
        for _t in (b'scan', b'odom', b'imu', b'battery', b'heartbeat'):
            self.sub_socket.setsockopt(zmq.SUBSCRIBE, _t)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1000)   # 1s timeout
        # Bound the receive queue to match the publisher, which already caps
        # its own side at 20 (jetson_robot_bridge.py: pub_socket.set_hwm(20)).
        # This side was left at ZMQ's default of 1000, so the two ends
        # disagreed by 50x and a Wi-Fi stall could bank seconds of telemetry
        # here instead of dropping it.
        #
        # Banked telemetry is worse than lost telemetry. _handle_scan stamps
        # each sweep on ARRIVAL, so a scan that sat in this queue for two
        # seconds is handed to AMCL labelled "now": the geometry is from where
        # the robot used to be, dated to where it is. AMCL cannot reconcile
        # that with its particles and throws the cloud across the map, which is
        # the half-second spin of the model and the scan together, followed by
        # a confident re-convergence onto the wrong pose.
        #
        # Must be set BEFORE connect() - ZMQ reads HWM when the pipe is built.
        self.sub_socket.setsockopt(zmq.RCVHWM, 20)
        telemetry_url = f"tcp://{self.jetson_ip}:{self.telemetry_port}"
        self.sub_socket.connect(telemetry_url)
        self.get_logger().info(f"Connected to Jetson telemetry at {telemetry_url}")

        # Command PUSH socket
        self.cmd_socket = self.zmq_context.socket(zmq.PUSH)
        cmd_url = f"tcp://{self.jetson_ip}:{self.cmd_port}"
        self.cmd_socket.connect(cmd_url)
        self.get_logger().info(f"Connected to Jetson command at {cmd_url}")

        # Auto-ARM upon startup
        if self.auto_arm:
            self._send_arm(True)
            self.get_logger().info("Auto-ARM enabled: Sent ARM command to Jetson on startup.")

        # Worker thread for receiving telemetry
        self.running = True
        self.recv_thread = threading.Thread(target=self._telemetry_worker, daemon=True)
        self.recv_thread.start()

    def _send_arm(self, arm: bool):
        try:
            payload = {'arm': bool(arm)}
            packed = msgpack.packb(payload, use_bin_type=True)
            self.cmd_socket.send(packed, flags=zmq.NOBLOCK)
            self.is_armed = bool(arm)
        except Exception as e:
            self.get_logger().warn(f"Failed to send ARM command to Jetson: {e}")

    def _arm_motors_cb(self, request, response):
        self._send_arm(request.data)
        response.success = True
        response.message = f"Sent ARM={request.data} command to Jetson."
        self.get_logger().info(response.message)
        return response

    def _cmd_vel_cb(self, msg: Twist):
        try:
            # Auto-arm if not yet armed and user wants to drive
            if self.auto_arm and not self.is_armed and (abs(msg.linear.x) > 0.001 or abs(msg.angular.z) > 0.001):
                self._send_arm(True)

            payload = {
                'linear_x': float(msg.linear.x),
                'angular_z': float(msg.angular.z)
            }
            packed = msgpack.packb(payload, use_bin_type=True)
            self.cmd_socket.send(packed, flags=zmq.NOBLOCK)
        except Exception as e:
            self.get_logger().warn(f"Failed to send cmd_vel to Jetson: {e}")

    def _telemetry_worker(self):
        while self.running and rclpy.ok():
            try:
                frames = self.sub_socket.recv_multipart()
                if len(frames) < 2:
                    continue
                topic, raw_payload = frames[0], frames[1]
                if topic not in (b'scan', b'odom', b'imu', b'battery', b'heartbeat'):
                    continue
                data = msgpack.unpackb(raw_payload, raw=False)

                if topic == b"scan":
                    self._handle_scan(data)
                elif topic == b"odom":
                    self._handle_odom(data)
                elif topic == b"imu":
                    self._handle_imu(data)
                elif topic == b"battery":
                    self._handle_battery(data)
                elif topic == b"heartbeat":
                    pass

            except zmq.Again:
                continue
            except Exception as e:
                if self.running:
                    self.get_logger().error(f"Telemetry decode error: {e}")
                time.sleep(0.01)

    def _sensor_stamp(self, data):
        """(stamp_seconds_on_this_clock, age_seconds) for one Jetson payload."""
        t_recv = self.get_clock().now().nanoseconds * 1e-9
        t_send = float(data.get('stamp', 0.0) or 0.0)
        if t_send <= 0.0:
            # Payload carries no stamp: fall back to arrival time, which is
            # what every message used to do.
            return t_recv, 0.0
        return self.jetson_clock.update(t_send, t_recv)

    def _too_stale(self, age, what):
        if self.max_sensor_age <= 0.0 or age <= self.max_sensor_age:
            return False
        self._stale_rejects.reject(
            '{0} arrived {1:.0f} ms late (limit {2:.0f} ms) - dropped'.format(
                what, age * 1000.0, self.max_sensor_age * 1000.0))
        return True

    def _handle_scan(self, data: dict):
        """Converts raw points [(angle_deg, dist_mm, intensity), ...] to LaserScan."""
        scan_msg = LaserScan()
        scan_msg.header.frame_id = self.laser_frame_id

        scan_msg.angle_min = 0.0
        # 720 bins = 0.5 deg. Measured on this unit: 492-504 points per
        # revolution at 600 rpm, i.e. one every 0.72 deg, so at this width
        # every return lands in a bin of its own - binning a real revolution
        # at 720 collided zero times.
        #
        # Do NOT "tighten" this to ~500 to match the point count. The empty
        # bins cost nothing: inf reads as "no return", and both nav2's costmaps
        # (inf_is_valid defaults false) and slam_toolbox skip it. At 500 bins
        # neighbouring returns start sharing one and the nearer-wins rule below
        # silently discards real measurements instead.
        bins = 720
        scan_msg.angle_increment = (2.0 * math.pi) / bins
        # One increment short of a full turn: bin 0 and a bin at exactly 2*pi
        # would be the same ray.
        scan_msg.angle_max = 2.0 * math.pi - scan_msg.angle_increment
        duration = float(data.get('duration', 0.1))
        scan_msg.time_increment = duration / bins
        scan_msg.scan_time = duration

        # LaserScan.header.stamp must be the time of the FIRST ray of the sweep.
        # The Jetson only ships a revolution once it is complete, so the moment
        # this callback runs is one whole revolution (~100 ms) plus a few ms of
        # Wi-Fi AFTER that first ray was measured. Stamping with "now" therefore
        # dated every scan ~100 ms too late, and AMCL and the costmaps then
        # transformed it using a TF from the wrong instant.
        #
        # Standing still that error is invisible. Rotating, it is omega * 0.1 s
        # of yaw: ~6 deg at 1.0 rad/s. That reads exactly like a lidar bolted on
        # crooked, except it comes and goes with the turn rate - which is why
        # the scan "sometimes" lines up with the map and sometimes does not.
        # The Jetson ships a revolution only once it is complete, so the
        # payload stamp is the LAST ray; one revolution back is the first one.
        # Measured on the Jetson's clock and translated here, so that Wi-Fi
        # jitter can no longer masquerade as sensor timing. See JetsonClock.
        stamp_s, age = self._sensor_stamp(data)
        if self._too_stale(age, 'scan'):
            return
        scan_msg.header.stamp = _stamp_from_seconds(stamp_s - duration)
        scan_msg.range_min = 0.05
        scan_msg.range_max = 12.0

        ranges = [float('inf')] * bins
        intensities = [0.0] * bins

        points = data.get('points', [])
        for angle_deg, dist_mm, intensity in points:
            if dist_mm <= 0:
                continue
            dist_m = dist_mm / 1000.0
            if dist_m < scan_msg.range_min or dist_m > scan_msg.range_max:
                continue

            # The LD19 reports its angle increasing CLOCKWISE, but a ROS
            # LaserScan is indexed counter-clockwise (REP-103: +X forward,
            # +Y left), so the raw angle has to be reversed here. The vendor
            # driver does exactly this in ldlidar_stl_ros2/src/demo.cpp, which
            # flips the index (beam_size - index - 1) whenever laser_scan_dir
            # is set - the option it logs as "Counterclockwise" and that
            # ld19.launch.py turns on. This path never did.
            #
            # A mirror is NOT a fixed yaw offset, which is why this hid for so
            # long: it reflects the scan about the robot's own heading. Parked,
            # the scan lines up perfectly against a map built the same way (it
            # matched room_map.pgm to 0.057 m), and every rotation of theta
            # then throws the scan off by 2*theta. Confirmed on this robot by
            # the cable sitting 0.085 m off the lidar on its LEFT, which was
            # being published on the right.
            ccw_deg = (360.0 - angle_deg) % 360.0
            bin_idx = int(ccw_deg * bins / 360.0) % bins
            # Keep nearer return
            if math.isinf(ranges[bin_idx]) or dist_m < ranges[bin_idx]:
                ranges[bin_idx] = dist_m
                intensities[bin_idx] = float(intensity)

        scan_msg.ranges = ranges
        scan_msg.intensities = intensities
        self.scan_pub.publish(scan_msg)

    def _handle_odom(self, data: dict):
        vx = data.get('vx', 0.0)
        wz = data.get('wz', 0.0)
        # twist.linear.x is the only field the EKF fuses from here.
        if not sensor_guard.odom_sample_ok(vx, wz,
                                           data.get('x', 0.0),
                                           data.get('y', 0.0),
                                           data.get('yaw', 0.0)):
            return self._odom_rejects.reject('odom payload outside physical range')

        odom_msg = Odometry()
        # The EKF integrates these, so jitter in the stamp becomes jitter in
        # the fused yaw. Same Jetson-clock translation as the scan.
        stamp_s, age = self._sensor_stamp(data)
        if self._too_stale(age, 'odom'):
            return
        stamp = _stamp_from_seconds(stamp_s)
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame_id
        odom_msg.child_frame_id = self.base_frame_id

        # Planar dead reckoning from wheel kinematics (Midpoint RK2)
        in_x = float(data.get('x', 0.0))
        in_y = float(data.get('y', 0.0))
        in_yaw = float(data.get('yaw', 0.0))

        if abs(in_x) > 1e-5 or abs(in_y) > 1e-5 or abs(in_yaw) > 1e-5:
            self.odom_x = in_x
            self.odom_y = in_y
            self.odom_yaw = in_yaw
        else:
            if self.last_odom_time is not None:
                dt = stamp_s - self.last_odom_time
                if 0.0 < dt < 0.5:
                    th_mid = self.odom_yaw + (float(wz) * dt * 0.5)
                    self.odom_x += float(vx) * math.cos(th_mid) * dt
                    self.odom_y += float(vx) * math.sin(th_mid) * dt
                    self.odom_yaw += float(wz) * dt
                    self.odom_yaw = math.atan2(math.sin(self.odom_yaw), math.cos(self.odom_yaw))
            self.last_odom_time = stamp_s

        now_m = time.monotonic()
        if now_m - getattr(self, '_last_odom_pub', 0.0) < 0.019:
            return
        self._last_odom_pub = now_m
        odom_msg.pose.pose.position.x = float(self.odom_x)
        odom_msg.pose.pose.position.y = float(self.odom_y)
        odom_msg.pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(float(self.odom_yaw))
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.covariance = _diag6(POSE_COV_DIAG)

        odom_msg.twist.twist.linear.x = float(data.get('vx', 0.0))
        odom_msg.twist.twist.linear.y = float(data.get('vy', 0.0))
        odom_msg.twist.twist.angular.z = float(data.get('wz', 0.0))
        odom_msg.twist.covariance = _diag6(TWIST_COV_DIAG)

        # The wheels are what license the ZUPT and the bias tracker.
        #
        # Use the per-wheel velocities the Jetson forwards rather than the vx/wz
        # it derived from them: those come straight off the STM32 as INTEGER
        # mm/s, so a stopped wheel is exactly 0 and "still" is exact. Testing
        # the derived floats instead would make standstill depend on rounding.
        if self.use_imu:
            left = data.get('left_vel_ms')
            right = data.get('right_vel_ms')
            if left is None or right is None:
                # Older Jetson bridge payloads carry only the derived velocities.
                moving = (abs(odom_msg.twist.twist.linear.x) > 1e-6
                          or abs(odom_msg.twist.twist.angular.z) > 1e-6)
            else:
                moving = (left != 0.0 or right != 0.0)
            self.gyro.set_wheel_motion(moving)

        self.odom_pub.publish(odom_msg)

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame_id
            t.child_frame_id = self.base_frame_id
            t.transform.translation.x = odom_msg.pose.pose.position.x
            t.transform.translation.y = odom_msg.pose.pose.position.y
            t.transform.translation.z = 0.0
            t.transform.rotation = odom_msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)

    def _handle_imu(self, data: dict):
        if not self.use_imu or self.imu_pub.get_subscription_count() == 0:
            return
        stamp_s, age = self._sensor_stamp(data)
        if self._too_stale(age, 'imu'):
            return
        imu_msg = Imu()
        imu_msg.header.stamp = _stamp_from_seconds(stamp_s)
        imu_msg.header.frame_id = self.imu_frame_id

        imu_msg.linear_acceleration.x = float(data.get('ax', 0.0))
        imu_msg.linear_acceleration.y = float(data.get('ay', 0.0))
        imu_msg.linear_acceleration.z = float(data.get('az', 9.81))

        # Bias, scale and ZUPT. Returns None while the initial bias calibration
        # is still running, and then NOTHING may be published: a biased vyaw in
        # those first seconds is exactly the error the EKF would integrate.
        if not sensor_guard.imu_sample_ok(
                data.get('ax', 0.0), data.get('ay', 0.0), data.get('az', 9.81),
                data.get('gx', 0.0), data.get('gy', 0.0), data.get('gz', 0.0)):
            return self._imu_rejects.reject('gyro/accel outside physical range')

        conditioned = self.gyro.condition(float(data.get('gz', 0.0)))
        if conditioned is None:
            return
        gz, yaw_var = conditioned

        imu_msg.angular_velocity.x = float(data.get('gx', 0.0))
        imu_msg.angular_velocity.y = float(data.get('gy', 0.0))
        imu_msg.angular_velocity.z = float(gz)

        # An all-zero covariance is not "unknown" to robot_localization - it
        # reads as "certain", which pins the yaw gain to 1.0 and lets a biased
        # gyro overrule the wheels outright.
        imu_msg.angular_velocity_covariance = self.gyro.angular_velocity_covariance(yaw_var)
        imu_msg.linear_acceleration_covariance = [
            0.04, 0.0, 0.0,
            0.0, 0.04, 0.0,
            0.0, 0.0, 0.04]

        # REP-145: a leading -1 declares that this message carries no absolute
        # orientation, which is true - there is no magnetometer.
        imu_msg.orientation_covariance = [-1.0] + [0.0] * 8
        self.imu_pub.publish(imu_msg)

    def _handle_battery(self, data: dict):
        now = time.monotonic()
        if now - getattr(self, '_last_batt', 0.0) < 0.5:
            return
        self._last_batt = now
        v = Float32()
        v.data = float(data.get('voltage', 0.0))
        self.volt_pub.publish(v)

        c = Float32()
        c.data = float(data.get('current', 0.0))
        self.curr_pub.publish(c)

    def destroy_node(self):
        self.running = False
        self.sub_socket.close()
        self.cmd_socket.close()
        self.zmq_context.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JetsonBridgeNode()
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
