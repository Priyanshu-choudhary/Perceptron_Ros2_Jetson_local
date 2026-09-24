#!/usr/bin/env python3
"""Rate and bandwidth of every topic, plus who publishes/subscribes.

Subscribes with raw=True (no deserialisation) so the audit itself stays cheap.
NOTE: subscribing can wake publishers that only publish when someone listens,
so this is run on its own, never during a CPU measurement window.
"""
import json
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rosidl_runtime_py.utilities import get_message

SKIP_PREFIX = ('/profiler',)


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    out = sys.argv[2] if len(sys.argv) > 2 else None
    rclpy.init()
    node = Node('topic_audit')
    time.sleep(2.0)
    stats = {}
    subs = []
    for name, types in node.get_topic_names_and_types():
        if name.startswith(SKIP_PREFIX) or not types:
            continue
        pubs = node.get_publishers_info_by_topic(name)
        subs_info = node.get_subscriptions_info_by_topic(name)
        stats[name] = {'type': types[0], 'n': 0, 'bytes': 0,
                       'publishers': sorted({p.node_name for p in pubs}),
                       'subscribers': sorted({s.node_name for s in subs_info if s.node_name != 'topic_audit'})}
        if not pubs:
            continue
        try:
            msg_t = get_message(types[0])
        except Exception:
            continue
        qos = QoSProfile(depth=50, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        def cb(msg, name=name):
            stats[name]['n'] += 1
            stats[name]['bytes'] += len(msg)
        subs.append(node.create_subscription(msg_t, name, cb, qos, raw=True))
    t0 = time.time()
    while time.time() - t0 < secs:
        rclpy.spin_once(node, timeout_sec=0.1)
    dt = time.time() - t0
    rows = []
    for name, s in stats.items():
        s['hz'] = s['n'] / dt
        s['kBps'] = s['bytes'] / dt / 1024
        rows.append((name, s))
    rows.sort(key=lambda r: -r[1]['hz'])
    for name, s in rows:
        print(f"{name:45s} {s['hz']:8.1f} Hz {s['kBps']:9.1f} kB/s  pub={','.join(s['publishers'])[:60]}  "
              f"subs={len(s['subscribers'])}:{','.join(s['subscribers'])[:80]}")
    if out:
        json.dump(stats, open(out, 'w'), indent=1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
