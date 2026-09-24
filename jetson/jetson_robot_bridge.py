#!/usr/bin/env python3
"""
High-Speed ZeroMQ Bridge for Jetson Nano -> PC ROS 2 Humble & Python Configurator
Streams LD19 LiDAR and STM32 Motor Driver / IMU to PC over local Wi-Fi / LAN.
Receives /cmd_vel and TinyFrame configuration commands with a safety watchdog.
Publishes topics on port 5555 (ZMQ PUB):
  - b"odom"      : {'stamp', 'timestamp_ms', 'vx', 'vy', 'wz', 'left_enc', 'right_enc', 'left_vel_ms', 'right_vel_ms'}
                   (wheel velocities only -- pose is estimated on the host)
  - b"battery"   : {'stamp', 'voltage', 'current', 'bus_raw', 'cur_raw'}
  - b"imu"       : {'stamp', 'timestamp_ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'}
  - b"scan"      : {'stamp', 'duration', 'rpm', 'points': [(angle, dist, intensity), ...]}
  - b"reply"     : raw bytes of 0x55 telemetry, 0x56 IMU, or 0xA5 TinyFrames (CONFIG_REPLY, ACK, etc.)
  - b"heartbeat" : {'seq', 'stamp', 'mock', 'lidar_port', 'stm32_port', 'aruco'}
  - b"aruco"     : {'stamp', 'seq', 'width', 'height', 'dict', 'ids', 'corners'}
                   ArUco CORNER PIXELS, one message per processed frame even when
                   nothing is seen. Pose is solved on the host -- see ArucoThread.
  - b"aruco_debug": raw JPEG bytes of the annotated frame. Off unless switched on.

Receives on port 5556 (ZMQ PULL):
  - msgpack dict: {'linear_x': float, 'angular_z': float}
  - msgpack dict: {'raw': bytes} (forwarded directly to STM32)
  - raw bytes starting with 0xA5 (forwarded directly to STM32)
  - msgpack dict: {'aruco_debug': bool} (toggle the b"aruco_debug" JPEG stream)
  - msgpack dict: {'aruco_enable': bool, 'hold': s, 'rate': Hz}
                  (--aruco-on-demand: switch the camera on for `hold` seconds;
                  resend to keep it on, False switches it off)
"""

import sys
import os
import time
import math
import struct
import logging
import threading
import argparse
import glob
from typing import Optional, List, Tuple

import zmq
import msgpack
import serial

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('JetsonBridge')

# ==============================================================================
# DRIVE GEOMETRY
# ==============================================================================
# Effective track width, used to turn the two wheel speeds into a yaw rate:
#     wz = (v_right - v_left) / WHEEL_BASE_EFFECTIVE_M
#
# This is NOT the geometric track (0.37762 m in the URDF). A 4-wheel skid-steer
# scrubs its tyres sideways through every turn, so it behaves as though the
# wheels were further apart than they are, and that gap widens as grip falls.
#
# 0.63 m was measured with odom_verification_test against 90 / 180 / 360 deg
# turns ON A MARBLE FLOOR AT 0.25 rad/s. It is only valid for those conditions:
# on carpet or rubber the scrub is smaller and this value will over-report
# every turn. Re-measure per surface, and per speed if you run much faster.
#
# The STM32 still mixes cmd_vel using CFG_WHEEL_BASE_M (0.43), so a commanded
# yaw rate is actually achieved at roughly 0.43 / 0.63 = 68% of what was asked
# until that constant is reflashed to match. Odometry reports the real rate
# either way, so the map stays correct - the robot just turns lazily.
WHEEL_BASE_EFFECTIVE_M = 0.63

# ==============================================================================
# LD19 / D500 LiDAR PROTOCOL
# ==============================================================================
LIDAR_HEADER = 0x54
LIDAR_VERLEN = 0x2C
LIDAR_PACKET_LEN = 47
LIDAR_POINT_COUNT = 12

CRC_TABLE = (
    0x00, 0x4d, 0x9a, 0xd7, 0x79, 0x34, 0xe3, 0xae,
    0xf2, 0xbf, 0x68, 0x25, 0x8b, 0xc6, 0x11, 0x5c,
    0xa9, 0xe4, 0x33, 0x7e, 0xd0, 0x9d, 0x4a, 0x07,
    0x5b, 0x16, 0xc1, 0x8c, 0x22, 0x6f, 0xb8, 0xf5,
    0x1f, 0x52, 0x85, 0xc8, 0x66, 0x2b, 0xfc, 0xb1,
    0xed, 0xa0, 0x77, 0x3a, 0x94, 0xd9, 0x0e, 0x43,
    0xb6, 0xfb, 0x2c, 0x61, 0xcf, 0x82, 0x55, 0x18,
    0x44, 0x09, 0xde, 0x93, 0x3d, 0x70, 0xa7, 0xea,
    0x3e, 0x73, 0xa4, 0xe9, 0x47, 0x0a, 0xdd, 0x90,
    0xcc, 0x81, 0x56, 0x1b, 0xb5, 0xf8, 0x2f, 0x62,
    0x97, 0xda, 0x0d, 0x40, 0xee, 0xa3, 0x74, 0x39,
    0x65, 0x28, 0xff, 0xb2, 0x1c, 0x51, 0x86, 0xcb,
    0x21, 0x6c, 0xbb, 0xf6, 0x58, 0x15, 0xc2, 0x8f,
    0xd3, 0x9e, 0x49, 0x04, 0xaa, 0xe7, 0x30, 0x7d,
    0x88, 0xc5, 0x12, 0x5f, 0xf1, 0xbc, 0x6b, 0x26,
    0x7a, 0x37, 0xe0, 0xad, 0x03, 0x4e, 0x99, 0xd4,
    0x7c, 0x31, 0xe6, 0xab, 0x05, 0x48, 0x9f, 0xd2,
    0x8e, 0xc3, 0x14, 0x59, 0xf7, 0xba, 0x6d, 0x20,
    0xd5, 0x98, 0x4f, 0x02, 0xac, 0xe1, 0x36, 0x7b,
    0x27, 0x6a, 0xbd, 0xf0, 0x5e, 0x13, 0xc4, 0x89,
    0x63, 0x2e, 0xf9, 0xb4, 0x1a, 0x57, 0x80, 0xcd,
    0x91, 0xdc, 0x0b, 0x46, 0xe8, 0xa5, 0x72, 0x3f,
    0xca, 0x87, 0x50, 0x1d, 0xb3, 0xfe, 0x29, 0x64,
    0x38, 0x75, 0xa2, 0xef, 0x41, 0x0c, 0xdb, 0x96,
    0x42, 0x0f, 0xd8, 0x95, 0x3b, 0x76, 0xa1, 0xec,
    0xb0, 0xfd, 0x2a, 0x67, 0xc9, 0x84, 0x53, 0x1e,
    0xeb, 0xa6, 0x71, 0x3c, 0x92, 0xdf, 0x08, 0x45,
    0x19, 0x54, 0x83, 0xce, 0x60, 0x2d, 0xfa, 0xb7,
    0x5d, 0x10, 0xc7, 0x8a, 0x24, 0x69, 0xbe, 0xf3,
    0xaf, 0xe2, 0x35, 0x78, 0xd6, 0x9b, 0x4c, 0x01,
    0xf4, 0xb9, 0x6e, 0x23, 0x8d, 0xc0, 0x17, 0x5a,
    0x06, 0x4b, 0x9c, 0xd1, 0x7f, 0x32, 0xe5, 0xa8,
)

def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = CRC_TABLE[crc ^ b]
    return crc


# ==============================================================================
# STM32 MOTOR DRIVER & IMU PROTOCOL
# ==============================================================================
PROTO_SOF_NEW    = 0xA5
PROTO_EOF_NEW    = 0x5A
TELEMETRY_HEADER = 0x55
TELEMETRY_END    = 0x0A
TELEMETRY_SIZE   = 27
TELEMETRY_FMT    = '<IHhiiiiBB'

IMU_HEADER       = 0x56
IMU_END          = 0x0A
IMU_SIZE         = 31
IMU_FMT          = '<IffffffBB'

FRAME_CMD_VEL    = 0x01

def calc_xor_checksum(data: bytes) -> int:
    cs = 0
    for b in data:
        cs ^= b
    return cs

def pack_cmd_vel(linear_x: float, angular_z: float) -> bytes:
    payload = struct.pack('<ff', float(linear_x), float(angular_z))
    length = len(payload)
    chk = calc_xor_checksum(bytes([FRAME_CMD_VEL, length]) + payload)
    return bytes([PROTO_SOF_NEW, FRAME_CMD_VEL, length]) + payload + bytes([chk, PROTO_EOF_NEW])

def bus_raw_to_volts(bus_raw: int) -> float:
    return ((bus_raw >> 3) * 4) / 1000.0

def cur_raw_to_amps(cur_raw: int) -> float:
    return (cur_raw * 0.4) / 1000.0


# ==============================================================================
# THREAD-SAFE ZEROMQ PUBLISHER
# ==============================================================================
class SafePublisher:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket
        self.lock = threading.Lock()

    def send(self, topic: bytes, payload: bytes):
        with self.lock:
            self.socket.send_multipart([topic, payload])


# ==============================================================================
# WORKER THREADS
# ==============================================================================
class LidarThread(threading.Thread):
    """Background thread reading LD19 serial LiDAR and publishing to ZMQ."""

    def __init__(self, port: str, baud: int, publisher: SafePublisher, mock: bool = False):
        super().__init__(daemon=True, name="LidarThread")
        self.port = port
        self.baud = baud
        self.publisher = publisher
        self.mock = mock
        self.running = True

        # Diagnostic and health tracking
        self.connected = False
        self.valid_packets = 0
        self.scan_count = 0
        self.last_points_count = 0
        self.last_rpm = 0.0
        self.last_rx_time = 0.0
        self.status_reason = "Initializing"
        self._last_err_log_time = 0.0
        self._first_packet_logged = False
        self._scans_in_window = 0
        self._last_window_time = time.time()
        self.scans_per_sec = 0.0

    def is_healthy(self) -> bool:
        if self.mock:
            return True
        return self.connected and (time.time() - self.last_rx_time < 2.0) and (self.valid_packets > 0)

    def run(self):
        logger.info(f"LiDAR thread started (port={self.port}, baud={self.baud}, mock={self.mock})")
        buf = bytearray()

        while self.running:
            if self.mock:
                time.sleep(0.1)
                now = time.time()
                fake_points = [
                    (round(deg, 2), int(1500 + 400 * math.sin(math.radians(deg * 4))), 200)
                    for deg in range(0, 360)
                ]
                scan_msg = {
                    'stamp': now,
                    'duration': 0.1,
                    'rpm': 600.0,
                    'points': fake_points
                }
                payload = msgpack.packb(scan_msg, use_bin_type=True)
                self.publisher.send(b"scan", payload)
                continue

            ser = None
            try:
                ser = serial.Serial(
                    self.port, self.baud, timeout=1.0,
                    bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                ser.reset_input_buffer()
                self.connected = True
                self.status_reason = "Port opened, awaiting LD19 data"
                logger.info(f"LiDAR serial port opened on {self.port} (baud={self.baud})")

                points = []
                last_angle = None
                t0 = time.time()

                while self.running:
                    chunk = ser.read(max(1, ser.in_waiting or 1))
                    if not chunk:
                        if time.time() - self.last_rx_time > 3.0 and self.valid_packets == 0:
                            self.status_reason = "No data bytes received (check power/cable)"
                        continue
                    buf.extend(chunk)

                    while len(buf) >= LIDAR_PACKET_LEN:
                        if buf[0] != LIDAR_HEADER or buf[1] != LIDAR_VERLEN:
                            del buf[0]
                            continue

                        frame = bytes(buf[:LIDAR_PACKET_LEN])
                        if crc8(frame[:LIDAR_PACKET_LEN - 1]) != frame[LIDAR_PACKET_LEN - 1]:
                            del buf[0]
                            continue

                        del buf[:LIDAR_PACKET_LEN]
                        self.valid_packets += 1
                        self.last_rx_time = time.time()
                        if not self._first_packet_logged:
                            self._first_packet_logged = True
                            logger.info(f"✓ [LIDAR CONNECTED & STREAMING] Received valid LD19 packet stream on {self.port}!")

                        radar_speed = struct.unpack_from('<H', frame, 2)[0]
                        start_angle = struct.unpack_from('<H', frame, 4)[0] * 0.01
                        end_angle = struct.unpack_from('<H', frame, 42)[0] * 0.01

                        span = (end_angle - start_angle) if end_angle >= start_angle else (end_angle + 360.0 - start_angle)
                        step = span / (LIDAR_POINT_COUNT - 1)

                        pkt_points = []
                        offset = 6
                        for i in range(LIDAR_POINT_COUNT):
                            dist = struct.unpack_from('<H', frame, offset)[0]
                            intensity = frame[offset + 2]
                            offset += 3
                            angle = (start_angle + i * step) % 360.0
                            # angle = (360.0 - angle) % 360.0
                            pkt_points.append((round(angle, 2), dist, intensity))

                        if last_angle is not None and start_angle < last_angle:
                            if len(points) >= 100:
                                now = time.time()
                                duration = now - t0
                                self.last_rpm = radar_speed * 60.0 / 360.0
                                self.scan_count += 1
                                self._scans_in_window += 1
                                self.last_points_count = len(points)
                                self.status_reason = "Streaming scans"

                                w_dur = now - self._last_window_time
                                if w_dur >= 2.0:
                                    self.scans_per_sec = self._scans_in_window / w_dur
                                    self._scans_in_window = 0
                                    self._last_window_time = now

                                scan_msg = {
                                    'stamp': now,
                                    'duration': duration,
                                    'rpm': self.last_rpm,
                                    'points': points
                                }
                                payload = msgpack.packb(scan_msg, use_bin_type=True)
                                self.publisher.send(b"scan", payload)
                                t0 = now
                            points = []

                        last_angle = start_angle
                        points.extend(pkt_points)

            except serial.SerialException as e:
                self.connected = False
                self.status_reason = f"Serial error: {e}"
                now = time.time()
                if now - self._last_err_log_time > 4.0:
                    self._last_err_log_time = now
                    logger.warning(f"LiDAR [{self.port}]: {e}. Check USB cable/hub connection.")
                time.sleep(2.0)
            except Exception as e:
                self.connected = False
                self.status_reason = f"Error: {e}"
                logger.error(f"LiDAR stream unexpected error: {e}")
                time.sleep(2.0)
            finally:
                self.connected = False
                if ser and ser.is_open:
                    ser.close()


class STM32WorkerThread(threading.Thread):
    """
    Subscribes to cmd_vel / configuration from PC (ZMQ PULL).
    Sends commands to STM32, and reads 0x55 telemetry / 0x56 IMU / 0xA5 TinyFrames.
    """

    def __init__(self, cmd_socket: zmq.Socket, publisher: SafePublisher, port: str = '/dev/ttyUSB0', baud: int = 115200, mock: bool = False,
                 aruco_thread=None, stale_timeout: float = 3.0):
        super().__init__(daemon=True, name="STM32Thread")
        self.cmd_socket = cmd_socket
        self.publisher = publisher
        self.aruco_thread = aruco_thread
        self.port = port
        self.baud = baud
        self.mock = mock
        self.running = True
        self.ser = None
        self.last_cmd_time = 0.0
        self.is_stopped = True
        # How long an OPEN port may stay silent before we assume the link is
        # wedged and force a reopen. See the watchdog in run() for why.
        self.stale_timeout = float(stale_timeout)
        self._port_open_time = 0.0

        # Diagnostic and health tracking
        self.connected = False
        self.telemetry_count = 0
        self.imu_count = 0
        self.reply_count = 0
        self.last_rx_time = 0.0
        self.last_voltage = 0.0
        self.last_current = 0.0
        self.last_left_enc = 0
        self.last_right_enc = 0
        self.status_reason = "Initializing"
        self._last_err_log_time = 0.0
        self._first_telemetry_logged = False
        self._first_imu_logged = False
        self._packets_in_window = 0
        self._last_window_time = time.time()
        self.hz = 0.0

    def is_healthy(self) -> bool:
        if self.mock:
            return True
        return self.connected and (time.time() - self.last_rx_time < 2.0) and (self.telemetry_count > 0 or self.imu_count > 0)

    def run(self):
        logger.info(f"STM32 thread started (port={self.port}, baud={self.baud}, mock={self.mock})")
        poller = zmq.Poller()
        poller.register(self.cmd_socket, zmq.POLLIN)

        read_buf = bytearray()

        while self.running:
            if self.mock:
                socks = dict(poller.poll(50))
                if self.cmd_socket in socks:
                    raw = self.cmd_socket.recv()
                    try:
                        data = msgpack.unpackb(raw, raw=False)
                        self.last_cmd_time = time.time()
                        lx = float(data.get("linear_x", 0.0))
                        az = float(data.get("angular_z", 0.0))
                        if abs(lx) > 0.001 or abs(az) > 0.001:
                            logger.info(f"[MOCK] Received cmd_vel: linear_x={lx:.2f} m/s, angular_z={az:.2f} rad/s")
                    except Exception:
                        pass

                now = time.time()
                odom_msg = {
                    'stamp': now, 'timestamp_ms': int(now * 1000),
                    'vx': 0.0, 'vy': 0.0, 'wz': 0.0,
                    'left_enc': 0, 'right_enc': 0,
                    'left_vel_ms': 0.0, 'right_vel_ms': 0.0
                }
                self.publisher.send(b"odom", msgpack.packb(odom_msg, use_bin_type=True))

                imu_msg = {
                    'stamp': now, 'timestamp_ms': int(now * 1000),
                    'ax': 0.0, 'ay': 0.0, 'az': 9.81,
                    'gx': 0.0, 'gy': 0.0, 'gz': 0.0
                }
                self.publisher.send(b"imu", msgpack.packb(imu_msg, use_bin_type=True))
                time.sleep(0.05)
                continue

            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
                self.ser.reset_input_buffer()
                self._port_open_time = time.time()
                self.connected = True
                self.status_reason = "Port opened, awaiting telemetry"
                logger.info(f"STM32 serial port opened on {self.port} (baud={self.baud})")

                while self.running:
                    # 1. Process incoming commands from PC
                    socks = dict(poller.poll(10))
                    if self.cmd_socket in socks:
                        raw = self.cmd_socket.recv()

                        # Case A: Direct TinyFrame raw bytes from PC
                        if raw.startswith(bytes([PROTO_SOF_NEW])):
                            self.last_cmd_time = time.time()
                            self.ser.write(raw)
                            logger.info(f"Forwarded raw TinyFrame ({len(raw)}B, type=0x{raw[1]:02X}) to STM32")
                        else:
                            # Case B: Msgpack payload
                            try:
                                data = msgpack.unpackb(raw, raw=False)
                                if "raw" in data and isinstance(data["raw"], (bytes, bytearray)):
                                    self.last_cmd_time = time.time()
                                    self.ser.write(data["raw"])
                                    logger.info(f"Forwarded raw bytes ({len(data['raw'])}B) to STM32")
                                elif "aruco_debug" in data:
                                    # The camera debug stream is toggled here
                                    # rather than on its own socket because
                                    # this is the only reader of cmd_socket;
                                    # a second consumer would steal cmd_vel.
                                    if self.aruco_thread is not None:
                                        self.aruco_thread.debug = bool(data["aruco_debug"])
                                        logger.info("ArUco debug JPEG stream %s",
                                                    "ON" if self.aruco_thread.debug else "OFF")
                                elif "aruco_enable" in data:
                                    # Camera on demand (docking). Deliberately
                                    # does NOT refresh last_cmd_time: a keepalive
                                    # for the camera must never keep the motor
                                    # watchdog from stopping a robot whose
                                    # cmd_vel source has died.
                                    if self.aruco_thread is not None:
                                        self.aruco_thread.request(
                                            bool(data["aruco_enable"]),
                                            hold=float(data.get("hold", 10.0)),
                                            rate=data.get("rate"))
                                elif "linear_x" in data or "angular_z" in data:
                                    self.last_cmd_time = time.time()
                                    lx = float(data.get("linear_x", 0.0))
                                    az = float(data.get("angular_z", 0.0))
                                    if abs(lx) > 0.001 or abs(az) > 0.001:
                                        self.is_stopped = False
                                    else:
                                        self.is_stopped = True
                                    frame = pack_cmd_vel(lx, az)
                                    self.ser.write(frame)
                            except Exception as e:
                                logger.warning(f"Error handling cmd payload: {e}")

                    # Safety Watchdog: If robot was moving and no cmd for > 500ms, send stop once
                    if not self.is_stopped and (time.time() - self.last_cmd_time > 0.5):
                        self.ser.write(pack_cmd_vel(0.0, 0.0))
                        self.is_stopped = True

                    # 2. Read telemetry & IMU from STM32
                    chunk = self.ser.read(max(1, self.ser.in_waiting or 1))
                    if chunk:
                        read_buf.extend(chunk)

                    # ── Stale-link watchdog ───────────────────────────────
                    # The reconnect path below only runs on a SerialException,
                    # but the most common real failure does not raise one: the
                    # FTDI bulk-in endpoint stalls (dmesg shows
                    # "usb_serial_generic_read_bulk_callback - urb stopped:
                    # -32", i.e. -EPIPE), usually when motor current browns out
                    # the USB hub. The port stays open and readable, read()
                    # just returns b'' forever. Without this the thread spins
                    # silently with connected=True and the PC sees telemetry,
                    # IMU and battery simply stop, with nothing logged anywhere
                    # -- which is exactly how a two-second glitch turns into a
                    # dead session.
                    # A longer grace before the FIRST frame: an ECU that is
                    # still booting must not be mistaken for a wedged link and
                    # reconnected in a tight loop.
                    silent_since = max(self.last_rx_time, self._port_open_time)
                    limit = (self.stale_timeout if self.last_rx_time > 0.0
                             else max(self.stale_timeout, 5.0))
                    if time.time() - silent_since > limit:
                        raise serial.SerialException(
                            "no valid frame for %.1fs on an open port; "
                            "link is wedged, reopening" % limit)

                    # ── Unified frame dispatcher ──────────────────────────
                    # Single loop that dispatches on the header byte. Each
                    # branch BREAKS (waits for more bytes) on a short frame
                    # instead of deleting, and only resyncs one byte at a time
                    # on a genuinely invalid frame. The previous version ran
                    # three separate `while len(buf) >= N` loops, so the 0x55
                    # pass consumed the buffer down to <27 bytes and the 0x56
                    # loop (needing 31) could never run — every IMU frame was
                    # shredded before it was ever parsed.
                    while read_buf:
                        head = read_buf[0]

                        # ---- 0x55 Unified Telemetry (27 B) ----
                        if head == TELEMETRY_HEADER:
                            if len(read_buf) < TELEMETRY_SIZE:
                                break
                            if read_buf[TELEMETRY_SIZE - 1] != TELEMETRY_END:
                                del read_buf[0]
                                continue

                            t_frame = bytes(read_buf[:TELEMETRY_SIZE])
                            if calc_xor_checksum(t_frame[:TELEMETRY_SIZE - 2]) != t_frame[TELEMETRY_SIZE - 2]:
                                del read_buf[0]
                                continue
                            del read_buf[:TELEMETRY_SIZE]

                            unpacked = struct.unpack_from(TELEMETRY_FMT, t_frame, 1)
                            # unpacked layout:
                            # 0: timestamp_ms (I)
                            # 1: bus_raw (H)
                            # 2: cur_raw (h)
                            # 3: left_encoder (i)
                            # 4: right_encoder (i)
                            # 5: left_velocity (i, mm/s)
                            # 6: right_velocity (i, mm/s)
                            now = time.time()
                            l_vel_ms = unpacked[5] / 1000.0
                            r_vel_ms = unpacked[6] / 1000.0
                            vx = (l_vel_ms + r_vel_ms) / 2.0
                            wz = (r_vel_ms - l_vel_ms) / WHEEL_BASE_EFFECTIVE_M

                            odom_msg = {
                                'stamp': now,
                                'timestamp_ms': unpacked[0],
                                'vx': float(vx),
                                'vy': 0.0,
                                'wz': float(wz),
                                'left_enc': int(unpacked[3]),
                                'right_enc': int(unpacked[4]),
                                'left_vel_ms': float(l_vel_ms),
                                'right_vel_ms': float(r_vel_ms)
                            }
                            self.publisher.send(b"odom", msgpack.packb(odom_msg, use_bin_type=True))

                            battery_msg = {
                                'stamp': now,
                                'voltage': bus_raw_to_volts(unpacked[1]),
                                'current': cur_raw_to_amps(unpacked[2]),
                                'bus_raw': int(unpacked[1]),
                                'cur_raw': int(unpacked[2])
                            }
                            self.publisher.send(b"battery", msgpack.packb(battery_msg, use_bin_type=True))
                            # Forward raw frame to reply topic for direct PC parsing
                            self.publisher.send(b"reply", t_frame)

                            self.telemetry_count += 1
                            self._packets_in_window += 1
                            self.last_rx_time = now
                            self.last_voltage = battery_msg['voltage']
                            self.last_current = battery_msg['current']
                            self.last_left_enc = odom_msg['left_enc']
                            self.last_right_enc = odom_msg['right_enc']
                            self.status_reason = "Telemetry active"

                            w_dur = now - self._last_window_time
                            if w_dur >= 2.0:
                                self.hz = self._packets_in_window / w_dur
                                self._packets_in_window = 0
                                self._last_window_time = now

                            if not self._first_telemetry_logged:
                                self._first_telemetry_logged = True
                                logger.info(f"✓ [STM32 CONNECTED & STREAMING] Telemetry active on {self.port}! (Batt: {self.last_voltage:.2f}V, Encoders: L={self.last_left_enc}, R={self.last_right_enc})")

                        # ---- 0x56 IMU Stream (31 B) ----
                        elif head == IMU_HEADER:
                            if len(read_buf) < IMU_SIZE:
                                break
                            if read_buf[IMU_SIZE - 1] != IMU_END:
                                del read_buf[0]
                                continue

                            i_frame = bytes(read_buf[:IMU_SIZE])
                            if calc_xor_checksum(i_frame[:IMU_SIZE - 2]) != i_frame[IMU_SIZE - 2]:
                                del read_buf[0]
                                continue
                            del read_buf[:IMU_SIZE]

                            unpacked = struct.unpack_from(IMU_FMT, i_frame, 1)
                            now = time.time()
                            imu_msg = {
                                'stamp': now,
                                'timestamp_ms': unpacked[0],
                                'ax': unpacked[1], 'ay': unpacked[2], 'az': unpacked[3],
                                'gx': unpacked[4], 'gy': unpacked[5], 'gz': unpacked[6]
                            }
                            self.publisher.send(b"imu", msgpack.packb(imu_msg, use_bin_type=True))
                            self.publisher.send(b"reply", i_frame)
                            self.imu_count += 1
                            self.last_rx_time = now

                            if not self._first_imu_logged:
                                self._first_imu_logged = True
                                logger.info(f"✓ [IMU STREAMING] 0x56 frames active on {self.port}! (ax={imu_msg['ax']:.2f}, ay={imu_msg['ay']:.2f}, az={imu_msg['az']:.2f} m/s2)")

                        # ---- 0xA5 TinyFrames (CONFIG_REPLY, ACK, IMU_CAL_REPLY) ----
                        elif head == PROTO_SOF_NEW:
                            if len(read_buf) < 5:
                                break
                            flen = read_buf[2]
                            total_len = 5 + flen
                            if len(read_buf) < total_len:
                                break
                            if read_buf[total_len - 1] != PROTO_EOF_NEW:
                                del read_buf[0]
                                continue

                            frame = bytes(read_buf[:total_len])
                            chk_bytes = bytes([read_buf[1], flen]) + frame[3:3 + flen]
                            if calc_xor_checksum(chk_bytes) != frame[total_len - 2]:
                                del read_buf[0]
                                continue

                            del read_buf[:total_len]
                            # Forward parsed TinyFrame to PC
                            self.publisher.send(b"reply", frame)
                            self.reply_count += 1
                            self.last_rx_time = time.time()
                            logger.info(f"Forwarded STM32 TinyFrame reply 0x{frame[1]:02X} ({len(frame)}B) to PC")

                        # ---- unknown byte: resync ----
                        else:
                            del read_buf[0]

            except serial.SerialException as e:
                self.connected = False
                self.status_reason = f"Serial error: {e}"
                now = time.time()
                if now - self._last_err_log_time > 4.0:
                    self._last_err_log_time = now
                    logger.warning(f"STM32 [{self.port}]: {e}. Check USB cable/hub connection.")
                time.sleep(2.0)
            except Exception as e:
                self.connected = False
                self.status_reason = f"Error: {e}"
                logger.error(f"STM32 serial unexpected error: {e}")
                time.sleep(2.0)
            finally:
                self.connected = False
                if self.ser and self.ser.is_open:
                    self.ser.close()


def _looks_like_stm32(buf: bytearray) -> bool:
    """True once >=2 XOR-valid 0x55/0x56 frames are seen in buf."""
    good = 0
    i = 0
    n = len(buf)
    while i < n:
        h = buf[i]
        if h == TELEMETRY_HEADER:
            size, end = TELEMETRY_SIZE, TELEMETRY_END
        elif h == IMU_HEADER:
            size, end = IMU_SIZE, IMU_END
        else:
            i += 1
            continue
        if i + size > n:
            break
        fr = buf[i:i + size]
        if fr[size - 1] == end and calc_xor_checksum(bytes(fr[:size - 2])) == fr[size - 2]:
            good += 1
            if good >= 2:
                return True
            i += size
        else:
            i += 1
    return False


def _looks_like_lidar(buf: bytearray) -> bool:
    """True once >=3 LD19 packets (0x54 0x2C, 47 B stride) are seen in buf."""
    good = 0
    i = 0
    n = len(buf)
    while i < n - 1:
        if buf[i] == 0x54 and buf[i + 1] == 0x2C:
            good += 1
            if good >= 3:
                return True
            i += 47
        else:
            i += 1
    return False


def _probe_port(port: str, baud: int, matcher, secs: float = 0.8) -> bool:
    """Open port at baud and see whether incoming bytes match the protocol."""
    ser = None
    try:
        ser = serial.Serial(port, baud, timeout=0.15)
        time.sleep(0.15)
        ser.reset_input_buffer()
        buf = bytearray()
        t0 = time.time()
        while time.time() - t0 < secs:
            chunk = ser.read(max(1, ser.in_waiting or 1))
            if chunk:
                buf.extend(chunk)
                if matcher(buf):
                    return True
        return False
    except Exception:
        return False
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass


def find_serial_ports(lidar_baud: int = 230400, stm32_baud: int = 115200) -> Tuple[str, str]:
    """Identify the LiDAR and STM32 ports by what each one is actually sending.

    /dev/ttyUSB* numbering is assigned in USB enumeration order, so it is NOT
    stable across a replug or reboot - the LiDAR and the STM32 TTL converter
    can and do swap places. The old version assumed STM32==ttyUSB0 and
    LiDAR==ttyUSB1, which silently fed LiDAR bytes to the STM32 parser (and
    vice versa) whenever they came up in the other order. Sniff instead.
    """
    candidates = sorted(set(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')))
    lidar_port = None
    stm32_port = None

    for port in candidates:
        if stm32_port is None and _probe_port(port, stm32_baud, _looks_like_stm32):
            stm32_port = port
            logger.info(f"Auto-detect: STM32 identified on {port} (valid 0x55/0x56 frames @ {stm32_baud})")
            continue
        if lidar_port is None and _probe_port(port, lidar_baud, _looks_like_lidar):
            lidar_port = port
            logger.info(f"Auto-detect: LiDAR identified on {port} (valid LD19 packets @ {lidar_baud})")
            continue

    # Fall back to the legacy positional guess for whatever stayed unidentified
    # (e.g. STM32 unpowered at startup) so behaviour degrades rather than crashes.
    if stm32_port is None:
        stm32_port = "/dev/ttyUSB0" if "/dev/ttyUSB0" in candidates else "/dev/ttyACM0"
        logger.warning(f"Auto-detect: no STM32 signature found; falling back to {stm32_port}")
    if lidar_port is None:
        remaining = [p for p in candidates if p != stm32_port]
        lidar_port = remaining[0] if remaining else "/dev/ttyUSB1"
        logger.warning(f"Auto-detect: no LiDAR signature found; falling back to {lidar_port}")

    return lidar_port, stm32_port


# ==============================================================================
# ARUCO MARKER DETECTION
# ==============================================================================
# WHY THIS THREAD SENDS CORNERS AND NOT A POSE
#
# Detection (adaptive threshold + quad fit + bit decode) is the expensive half
# and it needs the raw sensor pixels, so it has to happen here. solvePnP on 4
# points is free, so it does NOT have to happen here -- and it is much better
# off on the host, where the intrinsics, the marker->map database and the TF
# tree already live and can be changed without redeploying to the Jetson.
#
# This also matches what the rest of this bridge already does: it ships wheel
# velocities and lets the host integrate them. Same idea, same reason.
#
# WHY NOT SEND VIDEO AND DETECT ON THE PC
# ArUco accuracy is corner-localisation accuracy, and that is exactly what a
# lossy encoder destroys -- block artefacts land on the black/white border,
# which is the only thing carrying the signal. A 500 kbit/s H.265 stream costs
# roughly an order of magnitude of pose accuracy to save ~60 bytes per frame.
#
# RANGE, FOR THE 60 mm MARKERS THIS IS AIMED AT
# A marker of side S spans fx*S/Z pixels. With fx = 411 and S = 0.06 that is
# 24.7/Z px, and a 6X6 marker needs 8 modules across (6 data + 1 border each
# side). So ~3.1 px/module at 1 m and ~2.1 px/module at 1.5 m, which is about
# where decoding stops working. Treat this as a sub-1.5 m sensor.

ARUCO_DEFAULT_DICT = "DICT_6X6_250"


class ArucoThread(threading.Thread):
    """Detect ArUco markers on the USB camera; publish corner pixels only.

    Publishes b"aruco" once per processed frame, ALWAYS -- including when
    nothing was seen, so the host can tell "camera alive, no markers" from
    "camera dead". msgpack dict, self-describing by name rather than a packed
    struct, so adding a field cannot silently shear a consumer the way the
    0x55/0x56 serial frames can:

        {'stamp':   float,   time.time() sampled at frame grab
         'seq':     int,
         'width':   int,     pixels -- the host scales its intrinsics by this
         'height':  int,
         'dict':    str,     so a dictionary mismatch is caught, not debugged
         'ids':     [int, ...],
         'corners': [[x0,y0,x1,y1,x2,y2,x3,y3], ...]}

    Corner order is OpenCV's: top-left, top-right, bottom-right, bottom-left,
    in the marker's own frame. The host's object-point model must match.

    Optionally publishes b"aruco_debug" (JPEG bytes) when switched on at
    runtime with {'aruco_debug': true} on the command socket. Off by default:
    it is for finding out why detection is failing, and it is the one thing
    here that is not nearly free.

    ON DEMAND (always_on=False, --aruco-on-demand): the camera stays closed
    and OpenCV is not even imported until {'aruco_enable': true} arrives on
    the command socket, and it closes again when the request's `hold` runs
    out. The docking program resends the request every second, so a docking
    run that dies leaves the camera on for at most `hold` seconds. Idle cost
    is one sleeping thread; detection costs ~40 ms of one core per frame.
    """

    def __init__(self, publisher: SafePublisher, device: str = "/dev/video0",
                 width: int = 1280, height: int = 720, rate: float = 5.0,
                 dict_name: str = ARUCO_DEFAULT_DICT,
                 corner_refine_win: int = 3,
                 min_perimeter_rate: float = 0.01,
                 debug: bool = False, debug_rate: float = 2.0,
                 debug_quality: int = 60, always_on: bool = True):
        super().__init__(daemon=True, name="ArucoThread")
        self.publisher = publisher
        self.device = device
        self.width = width
        self.height = height
        self.period = 1.0 / rate if rate > 0 else 0.2
        self.default_period = self.period
        self.always_on = bool(always_on)
        self.demand_until = 0.0
        self.dict_name = dict_name
        self.corner_refine_win = max(2, int(corner_refine_win))
        self.min_perimeter_rate = float(min_perimeter_rate)
        self.debug = bool(debug)
        self.debug_period = 1.0 / debug_rate if debug_rate > 0 else 0.5
        self.debug_quality = int(debug_quality)

        self.running = True
        self.connected = False
        self.status_reason = "not started"
        self.seq = 0
        self.frames_processed = 0
        self.frames_with_markers = 0
        self.last_frame_time = 0.0

    def is_healthy(self) -> bool:
        if not self.wanted():
            return True
        return self.connected and (time.time() - self.last_frame_time < 3.0)

    def wanted(self) -> bool:
        return self.always_on or time.time() < self.demand_until

    def request(self, on: bool, hold: float = 10.0, rate=None):
        """{'aruco_enable': ...} from the command socket (on-demand mode)."""
        was = self.wanted()
        if on:
            self.demand_until = time.time() + max(1.0, min(float(hold), 120.0))
            try:
                r = float(rate) if rate is not None else 0.0
            except (TypeError, ValueError):
                r = 0.0
            self.period = 1.0 / min(r, 15.0) if r > 0 else self.default_period
        else:
            self.demand_until = 0.0
            self.period = self.default_period
        if was != self.wanted():
            logger.info("ArUco camera %s on request", "ON" if self.wanted() else "OFF")

    def status(self) -> dict:
        return {
            'connected': self.connected,
            'reason': self.status_reason,
            'frames': self.frames_processed,
            'hits': self.frames_with_markers,
            'debug': self.debug,
            'mode': 'on' if self.always_on else 'demand',
            'enabled': self.wanted(),
        }

    def _open_capture(self, cv2):
        """Open the camera as MJPG. Returns the capture, or None."""
        # CAP_V4L2 explicitly: the default backend here picks GStreamer and
        # then ignores FOURCC, which silently lands you on raw YUYV. At
        # 1280x720 that is ~1.3 MB/frame over USB 2.0 and the camera drops to
        # about 5 fps, which looks exactly like "detection is slow".
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Ask for the shallowest queue the driver will give us. Only a hint --
        # the grab() loop below is what actually keeps frames fresh.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _build_detector(self, cv2):
        aruco = cv2.aruco
        if not hasattr(aruco, self.dict_name):
            raise ValueError("unknown ArUco dictionary: " + self.dict_name)
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, self.dict_name))

        # OpenCV 4.5 on this Jetson has no aruco.ArucoDetector (4.7+); the
        # legacy free-function API is the only one available here.
        params = (aruco.DetectorParameters_create()
                  if hasattr(aruco, 'DetectorParameters_create')
                  else aruco.DetectorParameters())

        # Sub-pixel refinement is the difference between ~2 px and ~0.3 px of
        # corner noise, and corner noise is the entire error budget.
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        # The default 5 px window is wider than one module of a 60 mm 6X6
        # marker at 1 m (~3 px), so it drags the neighbouring module's edge
        # into the fit. Smaller window, more iterations, tighter epsilon.
        params.cornerRefinementWinSize = self.corner_refine_win
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
        # Default 0.03 of max(w,h) = 38 px perimeter = a 9.6 px marker. These
        # markers are small enough that the default sits uncomfortably close
        # to the cut; 0.01 keeps the far end of the range usable. Anything it
        # lets through that is not a marker still has to decode 36 bits.
        params.minMarkerPerimeterRate = self.min_perimeter_rate
        return dictionary, params

    def run(self):
        # On demand: nothing at all -- not even the cv2 import -- until the
        # first request.
        if not self.always_on:
            self.status_reason = "idle (on demand)"
            logger.info("ArUco on demand: camera closed until requested")
            while self.running and not self.wanted():
                time.sleep(0.2)
            if not self.running:
                return
        # Imported here, not at module scope: cv2 costs a couple of seconds
        # and a lot of RSS on a Nano, and a bridge run with --no-aruco or on a
        # box without OpenCV must still come up and drive the robot.
        try:
            import cv2
            import numpy as np
        except Exception as e:
            self.status_reason = "OpenCV unavailable: %s" % e
            logger.warning("ArUco disabled -- %s", self.status_reason)
            return

        try:
            dictionary, params = self._build_detector(cv2)
        except Exception as e:
            self.status_reason = "detector setup failed: %s" % e
            logger.error("ArUco disabled -- %s", self.status_reason)
            return

        logger.info("ArUco thread started (device=%s %dx%d @%.1f Hz, dict=%s)",
                    self.device, self.width, self.height, 1.0 / self.period,
                    self.dict_name)

        last_process = 0.0
        last_debug = 0.0

        while self.running:
            cap = None
            announced = False
            if not self.wanted():
                # On demand and not requested: camera closed, thread asleep.
                self.status_reason = "idle (on demand)"
                time.sleep(0.2)
                continue
            try:
                cap = self._open_capture(cv2)
                if cap is None:
                    self.connected = False
                    self.status_reason = "cannot open %s" % self.device
                    logger.warning("ArUco: %s (retry in 3s). Is something else "
                                   "holding the camera?", self.status_reason)
                    time.sleep(3.0)
                    continue

                self.connected = True
                self.status_reason = "capturing"

                while self.running and self.wanted():
                    # grab() dequeues without decoding, so spinning on it at
                    # the camera's own 30 fps drains the driver queue cheaply
                    # and the frame we finally decode is the newest one.
                    # Calling read() at 5 Hz instead hands back whatever was
                    # buffered several frames ago, and that staleness turns
                    # into pose error the moment the robot is turning.
                    if not cap.grab():
                        raise IOError("grab() failed -- camera went away?")

                    now = time.time()
                    if now - last_process < self.period:
                        continue
                    last_process = now

                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = cv2.aruco.detectMarkers(
                        gray, dictionary, parameters=params)

                    ids_out = []
                    corners_out = []
                    if ids is not None and len(ids) > 0:
                        for marker_id, quad in zip(ids.ravel(), corners):
                            flat = np.asarray(quad, dtype=np.float64).reshape(-1)
                            if flat.size != 8 or not np.isfinite(flat).all():
                                continue
                            ids_out.append(int(marker_id))
                            corners_out.append([float(v) for v in flat])

                    h, w = gray.shape[:2]
                    payload = {
                        'stamp': now,
                        'seq': self.seq,
                        'width': int(w),
                        'height': int(h),
                        'dict': self.dict_name,
                        'ids': ids_out,
                        'corners': corners_out,
                    }
                    self.publisher.send(b"aruco",
                                        msgpack.packb(payload, use_bin_type=True))
                    self.seq += 1
                    self.frames_processed += 1
                    self.last_frame_time = now
                    if ids_out:
                        self.frames_with_markers += 1

                    if not announced:
                        logger.info("[CAMERA CONNECTED & STREAMING] ArUco "
                                    "detection live on %s at %dx%d",
                                    self.device, w, h)
                        announced = True

                    if self.debug and (now - last_debug) >= self.debug_period:
                        last_debug = now
                        self._publish_debug(cv2, frame, corners, ids, now)

            except Exception as e:
                self.connected = False
                self.status_reason = str(e)
                logger.warning("ArUco: %s. Reopening in 2s.", e)
                time.sleep(2.0)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    self.connected = False

        self.connected = False
        self.status_reason = "stopped"
        logger.info("ArUco thread stopped.")

    def _publish_debug(self, cv2, frame, corners, ids, stamp):
        try:
            annotated = frame.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            label = "%s  n=%d  %s" % (
                self.dict_name, 0 if ids is None else len(ids),
                time.strftime("%H:%M:%S", time.localtime(stamp)))
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            ok, buf = cv2.imencode(
                '.jpg', annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.debug_quality])
            if ok:
                self.publisher.send(b"aruco_debug", buf.tobytes())
        except Exception as e:
            logger.debug("ArUco debug image failed: %s", e)


def main():
    detected_lidar, detected_stm32 = find_serial_ports()
    logger.info(f"Serial auto-detect: LiDAR={detected_lidar}  STM32={detected_stm32}")

    parser = argparse.ArgumentParser(description="Jetson Nano High-Speed Robot Bridge")
    parser.add_argument("--lidar-port", default=detected_lidar, help="LiDAR serial port")
    parser.add_argument("--lidar-baud", type=int, default=230400, help="LiDAR baud rate")
    parser.add_argument("--stm32-port", default=detected_stm32, help="STM32 serial port")
    parser.add_argument("--stm32-baud", type=int, default=115200, help="STM32 baud rate")
    parser.add_argument("--telemetry-port", type=int, default=5555, help="ZMQ PUB port")
    parser.add_argument("--cmd-port", type=int, default=5556, help="ZMQ PULL port")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without physical hardware")
    parser.add_argument("--no-aruco", action="store_true", help="Disable ArUco marker detection")
    parser.add_argument("--aruco-on-demand", action="store_true",
                        help="Camera closed until {'aruco_enable': true} arrives on the command port (docking)")
    parser.add_argument("--aruco-device", default="/dev/video0", help="Camera device for ArUco")
    parser.add_argument("--aruco-width", type=int, default=1280, help="Capture width (must match the calibration)")
    parser.add_argument("--aruco-height", type=int, default=720, help="Capture height (must match the calibration)")
    parser.add_argument("--aruco-rate", type=float, default=5.0, help="Detection rate in Hz")
    parser.add_argument("--aruco-dict", default=ARUCO_DEFAULT_DICT, help="ArUco dictionary name")
    parser.add_argument("--aruco-refine-win", type=int, default=3, help="Sub-pixel refinement window, px")
    parser.add_argument("--aruco-debug", action="store_true", help="Start with the debug JPEG stream on")
    args = parser.parse_args()

    context = zmq.Context()

    # ZMQ PUB for Telemetry (Dual-stack IPv4/IPv6)
    pub_socket = context.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.LINGER, 0)
    try:
        pub_socket.setsockopt(zmq.IPV6, 1)
    except Exception:
        pass
    pub_socket.set_hwm(20)
    pub_socket.bind(f"tcp://*:{args.telemetry_port}")
    logger.info(f"Telemetry PUB bound to tcp://*:{args.telemetry_port}")

    publisher = SafePublisher(pub_socket)

    # ZMQ PULL for Commands (/cmd_vel, Dual-stack IPv4/IPv6)
    cmd_socket = context.socket(zmq.PULL)
    cmd_socket.setsockopt(zmq.LINGER, 0)
    try:
        cmd_socket.setsockopt(zmq.IPV6, 1)
    except Exception:
        pass
    cmd_socket.bind(f"tcp://*:{args.cmd_port}")
    logger.info(f"Command PULL bound to tcp://*:{args.cmd_port}")

    # Start Worker Threads
    lidar_thread = LidarThread(args.lidar_port, args.lidar_baud, publisher, mock=args.mock)
    lidar_thread.start()

    aruco_thread = None
    if not args.no_aruco:
        aruco_thread = ArucoThread(
            publisher, device=args.aruco_device,
            width=args.aruco_width, height=args.aruco_height,
            rate=args.aruco_rate, dict_name=args.aruco_dict,
            corner_refine_win=args.aruco_refine_win, debug=args.aruco_debug,
            always_on=not args.aruco_on_demand)
        aruco_thread.start()

    # aruco_thread is handed over so {'aruco_debug': bool} on the command
    # socket can reach it; the STM32 thread is the only cmd_socket reader.
    stm32_thread = STM32WorkerThread(cmd_socket, publisher, args.stm32_port, args.stm32_baud,
                                     mock=args.mock, aruco_thread=aruco_thread)
    stm32_thread.start()

    logger.info("Jetson Robot Bridge active. Press Ctrl+C to stop.")
    try:
        seq = 0
        while True:
            time.sleep(1.0)
            seq += 1
            now = time.time()
            hb = {
                "seq": seq,
                "stamp": now,
                "mock": args.mock,
                "lidar_port": args.lidar_port,
                "stm32_port": args.stm32_port,
                # Thread health travels with the heartbeat so the host can tell
                # "the STM32 link died" from "the bridge died", instead of both
                # looking identical: topics that simply stop arriving.
                "stm32": {
                    "connected": stm32_thread.connected,
                    "healthy": stm32_thread.is_healthy(),
                    "reason": stm32_thread.status_reason,
                    "telemetry": stm32_thread.telemetry_count,
                    "imu": stm32_thread.imu_count,
                },
                "lidar": {
                    "connected": lidar_thread.connected,
                    "healthy": lidar_thread.is_healthy(),
                    "reason": lidar_thread.status_reason,
                    "packets": lidar_thread.valid_packets,
                },
                "aruco": aruco_thread.status() if aruco_thread else None
            }
            publisher.send(b"heartbeat", msgpack.packb(hb, use_bin_type=True))
    except KeyboardInterrupt:
        logger.info("Stopping Jetson Robot Bridge...")
    finally:
        lidar_thread.running = False
        stm32_thread.running = False
        if aruco_thread is not None:
            aruco_thread.running = False
        pub_socket.close()
        cmd_socket.close()
        context.term()
        logger.info("Bridge terminated.")


if __name__ == "__main__":
    main()
