#!/usr/bin/env python3
"""Python kernels shaped like the Python nodes (bridge scan/odom handling).
Runs on Python 3.6 (Jetson) and 3.10 (laptop); prints best-of-5 seconds."""
import json
import math
import random
import time


def best_of(n, f):
    b = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        b = min(b, time.perf_counter() - t0)
    return b


random.seed(1)
POINTS = [(random.uniform(0, 360), random.randint(0, 8000), random.randint(0, 255)) for _ in range(500)]


def scan_convert():
    # the loop body of jetson_bridge_node._handle_scan, 100 scans
    for _ in range(100):
        bins = 720
        ranges = [float('inf')] * bins
        inten = [0.0] * bins
        for angle_deg, dist_mm, intensity in POINTS:
            if dist_mm <= 0:
                continue
            d = dist_mm / 1000.0
            if d < 0.05 or d > 12.0:
                continue
            ccw = (360.0 - angle_deg) % 360.0
            i = int(ccw * bins / 360.0) % bins
            if math.isinf(ranges[i]) or d < ranges[i]:
                ranges[i] = d
                inten[i] = float(intensity)


def lidar_parse():
    # the per-packet unpack in jetson_robot_bridge.py LidarThread, 450 packets
    import struct
    frame = bytes(random.randint(0, 255) for _ in range(47))
    for _ in range(450):
        crc = 0
        for b in frame[:46]:
            crc = (crc ^ b) & 0xFF
            for _k in range(8):
                crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        start = struct.unpack_from('<H', frame, 4)[0] * 0.01
        pts = []
        off = 6
        for i in range(12):
            dist = struct.unpack_from('<H', frame, off)[0]
            off += 3
            pts.append((round((start + i * 0.8) % 360.0, 2), dist, frame[off - 1]))


def odom_math():
    x = y = th = 0.0
    for i in range(20000):
        dt = 0.011
        vx, wz = 0.3, 0.2
        m = th + wz * dt * 0.5
        x += vx * math.cos(m) * dt
        y += vx * math.sin(m) * dt
        th = math.atan2(math.sin(th + wz * dt), math.cos(th + wz * dt))
        cov = [0.0] * 36
        for k, v in enumerate((0.05, 0.05, 1e6, 1e6, 1e6, 0.1)):
            cov[k * 6 + k] = v
        d = {'vx': vx, 'wz': wz, 'x': x, 'y': y, 'yaw': th, 'stamp': i * dt}


print(json.dumps({'py_scan_convert': best_of(5, scan_convert),
                  'py_lidar_parse': best_of(5, lidar_parse),
                  'py_odom_math': best_of(5, odom_math)}))
