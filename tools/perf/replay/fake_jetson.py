#!/usr/bin/env python3
"""Local stand-in for jetson_robot_bridge.py, fed with REAL recorded payloads.

Schedule = what the unloaded Jetson was measured sending (pass 1, 100.8 Hz):
  every STM32 tick (100.8 Hz): odom, battery, reply(0x55 raw), imu, reply(0x56 raw)
  scan 10 Hz (cycling the recorded revolutions), heartbeat 1 Hz
Every msgpack 'stamp' is patched to time.time() at send, exactly as the Jetson
stamps a fresh sample. Commands on 5556 are drained and ignored.
"""
import pickle
import struct
import sys
import threading
import time

import zmq

frames = pickle.load(open(sys.argv[1], 'rb'))
TICK_HZ = float(sys.argv[2]) if len(sys.argv) > 2 else 100.8
SCAN_MAX_T = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
by = {}
t_first_scan = min(t for t, k, _ in frames if k == b'scan')
for t_rel, t, p in frames:
    # scans only from the clean stretch (a person walked past the robot ~40 s in)
    if t == b'scan' and t_rel - t_first_scan > SCAN_MAX_T:
        continue
    by.setdefault(t, []).append(p)
KEY = b'\xa5stamp\xcb'


def prep(p):
    i = p.find(KEY)
    return (bytearray(p), i + len(KEY) if i >= 0 else -1)


pools = {t: [prep(p) for p in by[t]] for t in by}
# reply frames alternate 0x55 (27 B) and 0x56 (31 B); keep them apart
r55 = [x for x in pools.get(b'reply', []) if x[0][:1] == b'\x55'] or pools[b'reply']
r56 = [x for x in pools.get(b'reply', []) if x[0][:1] == b'\x56'] or pools[b'reply']

ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.set_hwm(20)
pub.bind('tcp://127.0.0.1:5555')
pull = ctx.socket(zmq.PULL)
pull.bind('tcp://127.0.0.1:5556')


def drain():
    while True:
        pull.recv()


threading.Thread(target=drain, daemon=True).start()


def send(topic, item):
    buf, off = item
    if off >= 0:
        struct.pack_into('>d', buf, off, time.time())
    pub.send_multipart([topic, bytes(buf)])


k = {b'odom': 0, b'battery': 0, b'imu': 0, b'scan': 0, b'heartbeat': 0, 'r55': 0, 'r56': 0}


def nxt(name, pool):
    i = k[name] % len(pool)
    k[name] += 1
    return pool[i]


t0 = time.monotonic()
n_tick = n_scan = n_hb = 0
while True:
    now = time.monotonic() - t0
    due_tick = n_tick / TICK_HZ
    due_scan = n_scan / 10.0
    due_hb = float(n_hb)
    nxt_due = min(due_tick, due_scan, due_hb)
    if nxt_due > now:
        time.sleep(nxt_due - now)
        continue
    if due_tick <= now:
        send(b'odom', nxt(b'odom', pools[b'odom']))
        send(b'battery', nxt(b'battery', pools[b'battery']))
        send(b'reply', nxt('r55', r55))
        send(b'imu', nxt(b'imu', pools[b'imu']))
        send(b'reply', nxt('r56', r56))
        n_tick += 1
    if due_scan <= now:
        send(b'scan', nxt(b'scan', pools[b'scan']))
        n_scan += 1
    if due_hb <= now:
        send(b'heartbeat', nxt(b'heartbeat', pools[b'heartbeat']))
        n_hb += 1
