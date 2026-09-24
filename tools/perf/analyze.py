#!/usr/bin/env python3
"""Collapse results/<dir>/*.json into one summary JSON + a console table."""
import glob
import json
import os
import statistics
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'results', 'matrix')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, 'summary.json')

# group per-process labels into families so composition / renamed nodes compare
FAMILY = {
    'rviz2': 'RViz', 'aruco_localizer_node': 'ArUco localizer', 'jetson_bridge_node': 'Jetson bridge (ROS)',
    'battery_node': 'battery_node', 'amcl': 'AMCL', 'map_server': 'map_server',
    'controller_server': 'controller_server', 'planner_server': 'planner_server',
    'bt_navigator': 'bt_navigator', 'behavior_server': 'behavior_server',
    'smoother_server': 'smoother_server', 'velocity_smoother': 'velocity_smoother',
    'waypoint_follower': 'waypoint_follower', 'lifecycle_manager_navigation': 'lifecycle mgr (nav)',
    'lifecycle_manager_localization': 'lifecycle mgr (loc)', 'nav2_container': 'Nav2 container (all 8)',
    'robot_state_publisher': 'robot_state_publisher', 'joint_state_publisher': 'joint_state_publisher',
    'ekf_filter_node': 'EKF', 'ros2_launch': 'ros2 launch', 'path_overlay_node': 'path_overlay_node',
}


def fam(k):
    return FAMILY.get(k.split('#')[0], k.split('#')[0])


def phase(r, ph):
    p = r.get(ph)
    if not p:
        return None
    nodes = {}
    for k, v in p['procs'].items():
        f = fam(k)
        n = nodes.setdefault(f, {'cores': 0.0, 'pss_mb': 0.0, 'rss_mb': 0.0, 'p95': 0.0})
        n['cores'] += v['cores']
        n['p95'] += v['p95']
        n['pss_mb'] += (v['pss_kb'] or 0) / 1024
        n['rss_mb'] += (v['rss_kb'] or 0) / 1024
    out = {
        'cores': p['total_cores'], 'system_cores': p['system_busy_cores'], 'pss_mb': p['total_pss_mb'],
        'rss_mb': p['total_rss_mb'], 'nodes': nodes, 'seconds': p['seconds'],
        'jetson_hz': p.get('jetson_hz'),
    }
    if ph == 'nav':
        out['controller_hz'] = p['msgs']['cmd_vel_nav'] / p['seconds']
        out['smoother_hz'] = p['msgs']['cmd_vel'] / p['seconds']
        out['log'] = p.get('log', {})
        out['last_cmd'] = p.get('last_cmd')
    return out


def main():
    rows = {}
    for f in sorted(glob.glob(os.path.join(D, '*.json'))):
        if f.endswith('summary.json'):
            continue
        r = json.load(open(f))
        name = r['name']
        rows[name] = {'startup_s': r.get('startup_s'), 'error': r.get('error'),
                      'amcl_set_result': r.get('amcl_set_result'),
                      'launch_args': r.get('launch_args'), 'params_file': os.path.basename(r.get('params_file') or ''),
                      'amcl_set': r.get('amcl_set'),
                      'idle': phase(r, 'idle'), 'nav': phase(r, 'nav')}

    base_names = [n for n in rows if n.startswith(('A0_baseline', 'R1_', 'R2_', 'R3_')) and rows[n]['idle']]
    stats = {}
    for ph in ('idle', 'nav'):
        for key in ('cores', 'pss_mb', 'system_cores'):
            vals = [rows[n][ph][key] for n in base_names if rows[n][ph]]
            if vals:
                stats[f'{ph}_{key}'] = {'mean': statistics.mean(vals),
                                        'sd': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                                        'min': min(vals), 'max': max(vals), 'n': len(vals)}
        vals = [rows[n]['nav']['controller_hz'] for n in base_names if rows[n]['nav']]
        if vals and ph == 'nav':
            stats['nav_controller_hz'] = {'mean': statistics.mean(vals), 'sd': statistics.pstdev(vals),
                                          'n': len(vals)}
    # per-node baseline means
    node_base = {}
    for ph in ('idle', 'nav'):
        acc = {}
        for n in base_names:
            for k, v in (rows[n][ph] or {}).get('nodes', {}).items():
                acc.setdefault(k, []).append(v)
        node_base[ph] = {k: {'cores': statistics.mean(x['cores'] for x in v),
                             'cores_sd': statistics.pstdev([x['cores'] for x in v]) if len(v) > 1 else 0.0,
                             'pss_mb': statistics.mean(x['pss_mb'] for x in v)} for k, v in acc.items()}

    json.dump({'rows': rows, 'baseline': stats, 'baseline_names': base_names, 'node_baseline': node_base},
              open(OUT, 'w'), indent=1)

    bi = stats.get('idle_cores', {}).get('mean', 0)
    bn = stats.get('nav_cores', {}).get('mean', 0)
    print(f"baseline runs {base_names}")
    for k, v in stats.items():
        print(f"  {k:22s} mean {v['mean']:.3f} sd {v['sd']:.3f}" + (f" min {v['min']:.3f} max {v['max']:.3f}" if 'min' in v else ''))
    print(f"\n{'config':34s} {'idle':>6s} {'Δidle':>7s} {'nav':>6s} {'Δnav':>7s} {'ctrlHz':>6s} {'miss':>5s} {'PSS':>5s} {'odomHz':>6s}")
    for n, r in rows.items():
        i, v = r['idle'], r['nav']
        if not i:
            print(f"{n:34s} ERROR {r['error']}")
            continue
        print(f"{n:34s} {i['cores']:6.2f} {i['cores']-bi:+7.2f} "
              f"{(v or {}).get('cores', float('nan')):6.2f} {((v or {}).get('cores', float('nan'))-bn):+7.2f} "
              f"{(v or {}).get('controller_hz', float('nan')):6.1f} {(v or {}).get('log', {}).get('missed_control_rate', -1):5d} "
              f"{i['pss_mb']:5.0f} {i['jetson_hz']['odom'] if i.get('jetson_hz') else 0:6.1f}")


if __name__ == "__main__":
    main()
