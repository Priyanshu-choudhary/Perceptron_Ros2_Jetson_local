#!/usr/bin/env python3
"""robot cpu [SECONDS] - what every part of the stack costs, right now.

Exact CPU time from /proc/<pid>/stat over the window (100% = one core; the
Nano has 4), memory as PSS so the column adds up honestly. Runs on the
Jetson's own Python 3.6, no dependencies.
"""
import glob
import os
import re
import sys
import time

CLK = os.sysconf('SC_CLK_TCK')
ROOT = sys.argv[1] if len(sys.argv) > 1 else '/mnt/sdcard/ros2_chroot'
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0


def ticks(pid):
    with open('/proc/%s/stat' % pid) as f:
        s = f.read()
    fields = s[s.rfind(')') + 2:].split()
    return int(fields[11]) + int(fields[12])


def pss_kb(pid):
    total = 0
    try:
        with open('/proc/%s/smaps' % pid) as f:
            for line in f:
                if line.startswith('Pss:'):
                    total += int(line.split()[1])
    except (IOError, OSError):
        return 0
    return total


def label(pid):
    try:
        with open('/proc/%s/cmdline' % pid, 'rb') as f:
            cmd = f.read().replace(b'\0', b' ').decode(errors='replace')
    except (IOError, OSError):
        return None
    argv0 = os.path.basename(cmd.split(' ', 1)[0])
    if 'jetson_robot_bridge.py' in cmd:
        # the python process itself, not the sudo/env/setsid that started it
        return 'serial bridge (jetson_robot_bridge.py)' if argv0.startswith('python') else None
    m = re.search(r'__node:=(\w+)', cmd)
    if m:
        return m.group(1)
    if 'ros2 launch' in cmd:
        return 'ros2 launch'
    m = re.search(r'/lib/[\w]+/(\w+)', cmd)
    if m:
        return m.group(1)
    return None


def targets():
    out = {}
    for p in glob.glob('/proc/[0-9]*'):
        pid = p[6:]
        try:
            in_chroot = os.readlink(p + '/root') == ROOT
        except OSError:
            continue
        lab = label(pid)
        if lab and (in_chroot or lab.startswith('serial bridge')):
            out[pid] = lab
    return out


def board():
    with open('/proc/stat') as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]


t = targets()
if not t:
    print('nothing running')
    sys.exit(0)
t0 = {}
for pid in list(t):
    try:
        t0[pid] = ticks(pid)
    except (IOError, OSError):
        t.pop(pid)
b0 = board()
start = time.time()
time.sleep(SECS)
dt = time.time() - start
b1 = board()
rows = []
for pid, lab in t.items():
    try:
        rows.append((100.0 * (ticks(pid) - t0[pid]) / CLK / dt, pss_kb(pid) / 1024.0, lab))
    except (IOError, OSError):
        pass
ncpu = os.cpu_count() or 4
busy = (1.0 - (b1[1] - b0[1]) / max(1, b1[0] - b0[0])) * ncpu
rows.sort(reverse=True)
print('%-40s %8s %8s' % ('part (%.0f s window)' % dt, 'CPU %', 'PSS MB'))
for cpu, mem, lab in rows:
    print('%-40s %8.1f %8.0f' % (lab, cpu, mem))
print('-' * 58)
print('%-40s %8.1f %8.0f' % ('total of the above', sum(r[0] for r in rows), sum(r[1] for r in rows)))
print('%-40s %8.1f     (%.2f of %d cores busy, everything on the board)' % ('whole board', busy * 100, busy, ncpu))
with open('/proc/meminfo') as f:
    mi = dict((l.split(':')[0], int(l.split()[1])) for l in f)
print('memory available: %.0f MB of %.0f MB' % (mi['MemAvailable'] / 1024.0, mi['MemTotal'] / 1024.0))
