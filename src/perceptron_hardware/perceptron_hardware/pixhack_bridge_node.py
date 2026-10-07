import math
import os
import time
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from std_msgs.msg import Float32, String

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    mavutil = None
    PYMAVLINK_AVAILABLE = False


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """Convert Euler angles (rad) to quaternion (x, y, z, w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * cy - sr * sp * sy
    return x, y, z, w


class PixhackBridgeNode(Node):
    """
    Lightweight PyMAVLink bridge for Pixhack 2.4.8 FC.
    Publishes GPS (NavSatFix), IMU (sensor_msgs/Imu), and Compass Heading.
    Optimized for low CPU and memory footprint on Jetson Nano.
    """

    def __init__(self):
        super().__init__('pixhack_bridge_node')

        # Parameters
        self.declare_parameter('port', '/dev/pixhack')
        self.declare_parameter('fallback_ports', ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB2'])
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('gps_frame_id', 'gps_link')
        self.declare_parameter('publish_rate_hz', 50.0)

        self.port = self.get_parameter('port').value
        self.fallback_ports = self.get_parameter('fallback_ports').value
        self.baudrate = self.get_parameter('baudrate').value
        self.imu_frame_id = self.get_parameter('imu_frame_id').value
        self.gps_frame_id = self.get_parameter('gps_frame_id').value
        self.publish_rate_hz = self.get_parameter('publish_rate_hz').value

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Publishers
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', sensor_qos)
        self.heading_pub = self.create_publisher(Float32, '/compass/heading', 10)
        self.status_pub = self.create_publisher(String, '/pixhack/status', 5)

        # State cache
        self.master: Optional[mavutil.mavserial] = None
        self.connected = False
        self.running = True

        self.last_roll = 0.0
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.last_heading_deg = 0.0

        if not PYMAVLINK_AVAILABLE:
            self.get_logger().error(
                "pymavlink is NOT installed! Install it with: pip3 install pymavlink"
            )

        # Background thread for serial reading
        self.thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.thread.start()

        # Status watchdog timer
        self.create_timer(2.0, self._publish_status)

    def _find_available_port(self) -> Optional[str]:
        candidates = [self.port] + self.fallback_ports
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _connect(self) -> bool:
        if not PYMAVLINK_AVAILABLE:
            return False

        active_port = self._find_available_port()
        if not active_port:
            self.get_logger().warn(
                f"Pixhack port {self.port} not found (checked fallbacks {self.fallback_ports}). Retrying...",
                throttle_duration_sec=5.0
            )
            return False

        try:
            self.get_logger().info(f"Connecting to Pixhack on {active_port} at {self.baudrate} baud...")
            self.master = mavutil.mavlink_connection(
                active_port,
                baud=self.baudrate,
                autoreconnect=True
            )
            # Wait for heartbeat with timeout
            msg = self.master.wait_heartbeat(timeout=3.0)
            if msg:
                self.connected = True
                self.get_logger().info(
                    f"Pixhack connected! System: {self.master.target_system}, Component: {self.master.target_component}"
                )
                self._request_streams()
                return True
            else:
                self.get_logger().warn("Heartbeat timeout from Pixhack. Retrying...", throttle_duration_sec=5.0)
                return False
        except Exception as e:
            self.get_logger().error(f"Error opening connection to Pixhack: {e}", throttle_duration_sec=5.0)
            self.connected = False
            return False

    def _request_streams(self):
        """Request telemetry streams at required frequencies."""
        if not self.master:
            return
        try:
            # Request all streams at 20-50 Hz
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                int(self.publish_rate_hz),
                1
            )
            # Extra 1 (Attitude / IMU)
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
                50,
                1
            )
            # Position / GPS at 10 Hz
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                10,
                1
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to request MAVLink data streams: {e}")

    def _connection_loop(self):
        while self.running and rclpy.ok():
            if not self.connected or not self.master:
                if not self._connect():
                    time.sleep(2.0)
                    continue

            try:
                msg = self.master.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue

                msg_type = msg.get_type()
                if msg_type == 'ATTITUDE':
                    self._handle_attitude(msg)
                elif msg_type in ('GLOBAL_POSITION_INT', 'GPS_RAW_INT'):
                    self._handle_gps(msg)
                elif msg_type == 'RAW_IMU' or msg_type == 'SCALED_IMU':
                    self._handle_raw_imu(msg)
                elif msg_type == 'VFR_HUD':
                    self._handle_vfr_hud(msg)

            except Exception as e:
                self.get_logger().warn(f"MAVLink read error: {e}", throttle_duration_sec=5.0)
                self.connected = False
                time.sleep(1.0)

    def _handle_attitude(self, msg):
        self.last_roll = msg.roll
        self.last_pitch = msg.pitch
        self.last_yaw = msg.yaw

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self.imu_frame_id

        qx, qy, qz, qw = euler_to_quaternion(msg.roll, msg.pitch, msg.yaw)
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.orientation_covariance = [
            0.005, 0.0, 0.0,
            0.0, 0.005, 0.0,
            0.0, 0.0, 0.01
        ]

        imu_msg.angular_velocity.x = float(msg.rollspeed)
        imu_msg.angular_velocity.y = float(msg.pitchspeed)
        imu_msg.angular_velocity.z = float(msg.yawspeed)
        imu_msg.angular_velocity_covariance = [
            0.001, 0.0, 0.0,
            0.0, 0.001, 0.0,
            0.0, 0.0, 0.001
        ]

        # Linear acceleration filled if available
        imu_msg.linear_acceleration_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imu_pub.publish(imu_msg)

    def _handle_raw_imu(self, msg):
        # Convert mg (millig) or raw to m/s^2 (1g = 9.80665 m/s^2)
        # In ArduPilot RAW_IMU xacc is in mg
        ax = (msg.xacc / 1000.0) * 9.80665
        ay = (msg.yacc / 1000.0) * 9.80665
        az = (msg.zacc / 1000.0) * 9.80665

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self.imu_frame_id

        qx, qy, qz, qw = euler_to_quaternion(self.last_roll, self.last_pitch, self.last_yaw)
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw

        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az
        imu_msg.linear_acceleration_covariance = [
            0.05, 0.0, 0.0,
            0.0, 0.05, 0.0,
            0.0, 0.0, 0.05
        ]

        # Angular velocity from raw_imu if needed (xgyro is 10*rad/sec in ArduPilot raw_imu)
        imu_msg.angular_velocity.x = msg.xgyro / 1000.0
        imu_msg.angular_velocity.y = msg.ygyro / 1000.0
        imu_msg.angular_velocity.z = msg.zgyro / 1000.0

        self.imu_pub.publish(imu_msg)

    def _handle_gps(self, msg):
        gps_msg = NavSatFix()
        gps_msg.header.stamp = self.get_clock().now().to_msg()
        gps_msg.header.frame_id = self.gps_frame_id

        # lat/lon in 1e7 degrees
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = (msg.alt / 1000.0) if hasattr(msg, 'alt') else 0.0

        gps_msg.latitude = float(lat)
        gps_msg.longitude = float(lon)
        gps_msg.altitude = float(alt)

        # Fix status
        fix_type = getattr(msg, 'fix_type', 3)
        if fix_type >= 3:
            gps_msg.status.status = NavSatStatus.STATUS_FIX
        elif fix_type == 2:
            gps_msg.status.status = NavSatStatus.STATUS_FIX
        else:
            gps_msg.status.status = NavSatStatus.STATUS_NO_FIX

        gps_msg.status.service = NavSatStatus.SERVICE_GPS

        # Covariance based on HDOP if available
        eph = getattr(msg, 'eph', 200) / 100.0  # HDOP
        var = (eph * 2.5) ** 2
        gps_msg.position_covariance = [
            var, 0.0, 0.0,
            0.0, var, 0.0,
            0.0, 0.0, (var * 2.0)
        ]
        gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED

        self.gps_pub.publish(gps_msg)

    def _handle_vfr_hud(self, msg):
        heading_deg = float(msg.heading)
        self.last_heading_deg = heading_deg
        h_msg = Float32()
        h_msg.data = heading_deg
        self.heading_pub.publish(h_msg)

    def _publish_status(self):
        status_str = f"Pixhack: {'CONNECTED' if self.connected else 'DISCONNECTED'}, Heading: {self.last_heading_deg:.1f} deg"
        self.status_pub.publish(String(data=status_str))

    def destroy_node(self):
        self.running = False
        if self.master:
            try:
                self.master.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PixhackBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
