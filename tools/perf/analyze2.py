#!/usr/bin/env python3
"""Bracketed (A-B-A) analysis: every config vs the time-interpolated baseline."""
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import fam, phase  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'results', 'matrix2')
runs = {}
for f in glob.glob(os.path.join(D, '*.json')):
    if f.endswith('summary.json'):
        continue
    r = json.load(open(f))
    if not r.get('idle'):
        runs[r['name']] = {'error': r.get('error'), 'raw': r}
        continue
    runs[r['name']] = {'raw': r, 'idle': phase(r, 'idle'), 'nav': phase(r, 'nav'),
                       't_idle': (r['idle']['t_start'] + r['idle']['t_end']) / 2,
                       't_nav': ((r['nav']['t_start'] + r['nav']['t_end']) / 2) if r.get('nav') else None,
                       'startup_s': r.get('startup_s'), 'amcl_set_result': r.get('amcl_set_result')}

probe = []
pp = os.path.join(D, 'probe.log')
if os.path.exists(pp):
    for line in open(pp):
        try:
            t, ms = line.split()
            probe.append((float(t), float(ms)))
        except ValueError:
            pass


def speed(t0, t1):
    v = [ms for t, ms in probe if t0 <= t <= t1]
    return statistics.median(v) if v else None


bases = sorted([n for n in runs if n.startswith('BASE_') and 'idle' in runs[n]], key=lambda n: runs[n]['t_idle'])


def interp(ph, t, getter):
    """Linear interpolation of a baseline quantity at time t."""
    pts = [(runs[b]['t_' + ph], getter(runs[b][ph])) for b in bases if runs[b][ph]]
    pts = [(x, y) for x, y in pts if y is not None]
    before = [p for p in pts if p[0] <= t]
    after = [p for p in pts if p[0] >= t]
    if before and after:
        (x0, y0), (x1, y1) = before[-1], after[0]
        if x1 == x0:
            return y0, (before[-1], after[0])
        return y0 + (y1 - y0) * (t - x0) / (x1 - x0), (before[-1], after[0])
    p = (before or after)
    return p[-1][1] if before else p[0][1], None


def node_val(nodes, k):
    return nodes.get(k, {'cores': 0.0})['cores']


rows = {}
for n, r in runs.items():
    if 'idle' not in r:
        rows[n] = {'error': r['error']}
        continue
    row = {'startup_s': r['startup_s'], 'amcl_set_result': r['amcl_set_result'],
           'launch_args': r['raw'].get('launch_args'), 'params_file': os.path.basename(r['raw'].get('params_file') or ''),
           'amcl_set': r['raw'].get('amcl_set')}
    for ph in ('idle', 'nav'):
        p = r[ph]
        if not p:
            row[ph] = None
            continue
        t = r['t_' + ph]
        bcores, _ = interp(ph, t, lambda x: x['cores'])
        bpss, _ = interp(ph, t, lambda x: x['pss_mb'])
        allnodes = set(p['nodes'])
        for b in bases:
            if runs[b][ph]:
                allnodes |= set(runs[b][ph]['nodes'])
        nd = {}
        for k in allnodes:
            bv, _ = interp(ph, t, lambda x, k=k: node_val(x['nodes'], k))
            nd[k] = {'cores': node_val(p['nodes'], k), 'base': bv,
                     'pss_mb': p['nodes'].get(k, {}).get('pss_mb', 0.0)}
        d = dict(p)
        d.update({'base_cores': bcores, 'delta': p['cores'] - bcores, 'rel': p['cores'] / bcores - 1,
                  'base_pss': bpss, 'delta_pss': p['pss_mb'] - bpss, 'nodes': nd,
                  'probe_ms': speed(r['raw'][ph]['t_start'], r['raw'][ph]['t_end'])})
        if ph == 'nav':
            bhz, _ = interp(ph, t, lambda x: x['controller_hz'])
            d['base_controller_hz'] = bhz
        row[ph] = d
    rows[n] = row

# noise: successive baseline differences (includes ~4-5 min of drift)
noise = {}
for ph in ('idle', 'nav'):
    vals = [runs[b][ph]['cores'] for b in bases if runs[b][ph]]
    diffs = [abs(b - a) for a, b in zip(vals, vals[1:])]
    noise[ph] = {'values': vals, 'mean': statistics.mean(vals) if vals else None,
                 'sd': statistics.pstdev(vals) if len(vals) > 1 else None,
                 'successive_diff_mean': statistics.mean(diffs) if diffs else None,
                 'single_run_sd': (statistics.pstdev(
                     [b - a for a, b in zip(vals, vals[1:])]) / math.sqrt(2)) if len(diffs) > 1 else None}
# baseline per-node means
node_base = {}
for ph in ('idle', 'nav'):
    acc = {}
    for b in bases:
        for k, v in (runs[b][ph] or {}).get('nodes', {}).items():
            acc.setdefault(k, []).append(v)
    node_base[ph] = {k: {'cores': statistics.mean(x['cores'] for x in v),
                         'cores_sd': statistics.pstdev([x['cores'] for x in v]) if len(v) > 1 else 0.0,
                         'pss_mb': statistics.mean(x['pss_mb'] for x in v), 'n': len(v)} for k, v in acc.items()}
hz = [runs[b]['nav']['controller_hz'] for b in bases if runs[b]['nav']]
out = {'rows': rows, 'bases': bases, 'noise': noise, 'node_baseline': node_base,
       'baseline': {'idle_cores': noise['idle'], 'nav_cores': noise['nav'],
                    'idle_pss_mb': statistics.mean(runs[b]['idle']['pss_mb'] for b in bases),
                    'nav_controller_hz': statistics.mean(hz) if hz else None},
       'probe': probe}
json.dump(out, open(os.path.join(D, 'summary.json'), 'w'), indent=1)

print('baselines:', ', '.join(f"{b}={runs[b]['idle']['cores']:.2f}/{(runs[b]['nav'] or {}).get('cores', 0):.2f}"
                              f"@{speed(runs[b]['raw']['idle']['t_start'], runs[b]['raw']['idle']['t_end']) or 0:.2f}ms"
                              for b in bases))
print('noise:', {ph: {k: (round(v, 3) if isinstance(v, float) else None) for k, v in noise[ph].items() if k != 'values'}
                 for ph in noise})
print(f"\n{'config':36s} {'idle':>5s} {'Δidle':>6s} {'rel':>5s} {'nav':>5s} {'Δnav':>6s} {'rel':>5s} "
      f"{'ctrlHz':>6s} {'bHz':>5s} {'miss':>4s} {'PSS':>4s} {'ΔPSS':>5s} {'probe':>5s}")
for n in sorted(rows, key=lambda n: runs[n].get('t_idle', 0)):
    r = rows[n]
    if 'error' in r and r.get('idle') is None:
        print(f'{n:36s} ERROR {r["error"]}')
        continue
    i, v = r['idle'], r['nav'] or {}
    print(f"{n:36s} {i['cores']:5.2f} {i['delta']:+6.2f} {i['rel']*100:+4.0f}% {v.get('cores', 0):5.2f} "
          f"{v.get('delta', 0):+6.2f} {v.get('rel', 0)*100:+4.0f}% {v.get('controller_hz', 0):6.1f} "
          f"{v.get('base_controller_hz') or 0:5.1f} {v.get('log', {}).get('missed_control_rate', 0):4d} "
          f"{i['pss_mb']:4.0f} {i['delta_pss']:+5.0f} {i['probe_ms'] or 0:5.2f}")
