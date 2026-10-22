"""Stage 9 scenario suite: the scenes, the metrics taken from the evaluators, the limits (no ROS).

Each scenario is a bridge command line, the stack to run against it, how long to score it
(sim seconds, after a warm-up), and the limits its metrics must meet. The runner
(scenario_suite.py) starts everything, collects the evaluators' JSON summaries and the
topic rates, and checks them here: a regression shows up as a FAIL row, not as a feeling
that RViz looks worse.

Limits are the measured values of the reference run (README, Stage 9) with a margin, so a
pass means "no worse than when the limit was set", not "good enough for the road".
"""

import math

import numpy as np

# ---------------------------------------------------------------- the scenes
# bridge: carla_bridge arguments (seeded: same spawn points, traffic and walkers each run)
# stack:  'perception' = Stages 4-8 + their evaluators, 'localization' = Stage 3 EKFs + evaluator
SCENARIOS = {
    'static_props': {
        'about': 'cones and boxes at 5-40 m, a parked car; the ego drives past at 1 m/s '
                 '(static objects must stay static while the robot moves)',
        'bridge': '--scenario --walkers 0 --ego-speed 1.0 --seed 7',
        'stack': 'perception', 'duration': 35.0,
        'limits': [('fusion.other.range_mae', '<=', 0.4),
                   ('tracking.other.v_static_mae', '<=', 0.15),
                   ('costmap.recall.vehicle_static', '>=', 90.0),
                   ('costmap.recall.other_static', '>=', 80.0),
                   ('costmap.false_obstacle_pct', '<=', 3.0),
                   ('costmap.flicker_pct', '<=', 0.5)],
    },
    'crossing': {
        'about': '6 pedestrians pacing across the road 5-30 m ahead at 0.8-1.4 m/s, ego still',
        'bridge': '--crossing 6 --seed 7',
        'stack': 'perception', 'duration': 45.0,
        'limits': [('fusion.pedestrian.coverage_pct', '>=', 60.0),
                   ('fusion.pedestrian.range_mae', '<=', 0.35),
                   ('tracking.pedestrian.pos_mae', '<=', 0.35),
                   ('tracking.pedestrian.v_moving_mae', '<=', 0.35),
                   ('prediction.pedestrian_moving.err_2s', '<=', 1.3),
                   ('prediction.pedestrian_moving.in95_2s', '>=', 80.0),
                   ('costmap.recall.pedestrian_moving', '>=', 95.0),
                   ('costmap.dynamic_2s_pct', '>=', 85.0),
                   ('costmap.false_obstacle_pct', '<=', 3.0)],
    },
    'occluded_crossing': {
        'about': 'a van parked 6 m ahead on the left; 4 pedestrians cross at 10-25 m and '
                 'step out from behind it',
        'bridge': '--crossing 4 --occluder --seed 7',
        'stack': 'perception', 'duration': 45.0,
        'limits': [('tracking.pedestrian.pos_mae', '<=', 0.45),
                   ('tracking.pedestrian.id_switches_per_100', '<=', 2.0),
                   ('costmap.recall.pedestrian_moving', '>=', 90.0),
                   ('costmap.recall.vehicle_static', '>=', 90.0),
                   ('costmap.dynamic_2s_pct', '>=', 70.0)],
    },
    'city': {
        'about': 'autopilot through Town10 with 40 cars and 80 pedestrians: moving and stopped '
                 'cars, crowds, natural occlusion',
        'bridge': '--autopilot --traffic 40 --pedestrians 80 --seed 3',
        'stack': 'perception', 'duration': 90.0,
        'limits': [('cloud.ground_precision', '>=', 94.5),
                   ('cloud.ground_recall', '>=', 99.0),
                   ('fusion.vehicle.range_mae', '<=', 1.6),
                   # the lidar hits almost no sidewalk pedestrian here (README, Stage 9): guard
                   # the error of the ranges the fusion still gives them (12.4 m before the size gate)
                   ('fusion.pedestrian.range_mae', '<=', 8.0),
                   ('tracking.vehicle.v_moving_mae', '<=', 2.0),
                   ('costmap.recall.vehicle_moving', '>=', 95.0),
                   ('costmap.recall.vehicle_static', '>=', 95.0),
                   ('costmap.false_obstacle_pct', '<=', 4.0),
                   ('costmap.ms_p95', '<=', 80.0)],
    },
    'turns': {
        'about': 'autopilot forced through left and right turns at the next junctions, '
                 '15 cars (turning ego: the hard case for ego-motion and prediction)',
        'bridge': '--autopilot --traffic 15 --route Left,Right,Left,Right,Left --seed 5',
        'stack': 'perception', 'duration': 90.0,
        'limits': [('cloud.ground_precision', '>=', 94.5),
                   ('tracking.vehicle.v_static_mae', '<=', 1.0),
                   ('costmap.recall.vehicle_static', '>=', 90.0),
                   ('costmap.false_obstacle_pct', '<=', 4.0),
                   ('costmap.flicker_pct', '<=', 0.5)],
    },
    'dropout': {
        'about': 'city traffic; the LiDAR goes silent for 2 s at 30 s and the camera for 2 s '
                 'at 50 s: the stack must ride through and recover',
        'bridge': '--autopilot --traffic 30 --pedestrians 40 --seed 4 --dropout lidar:30:2,camera:50:2',
        'stack': 'perception', 'duration': 75.0,
        # recall: since the Stage 11 collision sensor joined the actors, Traffic Manager sends the ego
        # another way in this seed (red light by a junction); moving-car recall is 81.3 % here with
        # AND without the dropouts, so the dropout costs nothing and the limit is that scene's level
        'limits': [('health.nodes_missing', '==', 0),
                   ('health.outputs_after_dropout', '==', 1),
                   ('costmap.recall.vehicle_moving', '>=', 78.0),
                   ('costmap.false_obstacle_pct', '<=', 4.0)],
    },
    'localization': {
        'about': 'the Stage 3 EKFs (fixes on) on simulated wheel odometry, IMU and GNSS; '
                 'autopilot with 10 cars',
        'bridge': '--autopilot --tf-mode ekf --traffic 10 --seed 6',
        'stack': 'localization', 'duration': 120.0,
        # the global EKF is the one non-deterministic result in the suite: same seed, RMSE
        # 0.56 / 0.56 / 1.96 m over three runs (navsat's datum depends on message timing)
        'limits': [('loc.global.pos_rmse_m', '<=', 3.0),
                   ('loc.global.pos_max_m', '<=', 5.0),
                   ('loc.global.head_rmse_deg', '<=', 5.0),
                   ('loc.local.samples', '>=', 100)],
    },
}

# Stage 11: Nav2 drives the car (bridge --control ros) to a GPS goal; no traffic rules.
# goal: {'road': [lo, hi]} = a road point whose road route from the start is lo-hi m
#       (seeded pick; 'forward': only points ahead of the car), or {'ahead': d} = d m ahead.
# Pedestrian clearance only has to stay above "touching" (5 cm): the crossing scenes' walkers are
# steered blind (no AI), and one that turns at the kerb can walk into the side of a car already
# passing it, which no speed limit undoes. Near passes (< 0.5 m) are counted in
# nav.near_misses_pedestrian and listed in the notes, not hidden.
NAV_LIMITS = [('nav.reached', '==', 1), ('nav.collisions', '==', 0)]
SCENARIOS.update({
    'nav_open': {
        'about': 'Nav2 to a GPS goal 150-250 m away by road, round corners, in an empty Town10 '
                 '(no cars, no people)',
        'bridge': '--control ros --seed 7',
        'stack': 'navigation', 'duration': 260.0, 'goal': {'road': [150.0, 250.0]},
        'limits': NAV_LIMITS + [('nav.time_s', '<=', 220.0)],
    },
    'nav_cars': {
        'about': 'the same kind of goal with 30 cars driving on autopilot (they keep traffic rules, '
                 'the ego does not)',
        'bridge': '--control ros --traffic 30 --seed 7',
        'stack': 'navigation', 'duration': 300.0, 'goal': {'road': [150.0, 250.0]},
        'limits': NAV_LIMITS + [('nav.min_clearance_vehicle', '>=', 0.3)],
    },
    'nav_people': {
        'about': 'a goal 60 m straight ahead through 6 pedestrians pacing across the road, '
                 'and 60 more walking around',
        'bridge': '--control ros --crossing 6 --pedestrians 60 --seed 7',
        'stack': 'navigation', 'duration': 200.0, 'goal': {'ahead': 60.0},
        'limits': NAV_LIMITS + [('nav.min_clearance_pedestrian', '>=', 0.05)],
    },
    'nav_mixed': {
        'about': 'cars and people together: 30 cars, 80 pedestrians, 4 crossing ahead; a road goal '
                 '150-250 m away in front of the car',
        'bridge': '--control ros --traffic 30 --pedestrians 80 --crossing 4 --seed 7',
        'stack': 'navigation', 'duration': 320.0, 'goal': {'road': [150.0, 250.0], 'forward': True},
        'limits': NAV_LIMITS + [('nav.min_clearance_pedestrian', '>=', 0.05),
                                ('nav.min_clearance_vehicle', '>=', 0.3)],
    },
})

# every scenario also checks these (when the metric exists for its stack)
COMMON_LIMITS = [('health.nodes_missing', '==', 0),
                 ('rate.realtime_factor', '>=', 0.3)]

# the evaluator nodes each stack runs; their JSON lands in <out>/<name>.json
EVALUATORS = {
    'perception': ['cloud_eval_node', 'fusion_eval_node', 'tracking_eval_node',
                   'prediction_eval_node', 'costmap_eval_node'],
    'localization': ['localization_eval_node'],
    'navigation': [],
}
STACK_NODES = {
    'perception': ['lidar_preprocessor_node', 'object_fusion_node', 'object_tracker_node',
                   'trajectory_predictor_node', 'costmap_node'] + EVALUATORS['perception'],
    'localization': ['ekf_filter_node_local', 'ekf_filter_node_global', 'navsat_transform_node',
                     'localization_eval_node'],
    'navigation': ['lidar_preprocessor_node', 'object_fusion_node', 'object_tracker_node',
                   'trajectory_predictor_node', 'costmap_node', 'controller_server', 'planner_server',
                   'behavior_server', 'bt_navigator', 'gps_goal_node', 'collision_monitor',
                   'predictive_speed_limit_node'],
}

# topics whose rates the runner measures (sim-time Hz)
RATE_TOPICS = {
    'lidar': '/carla/lidar/points', 'camera': '/carla/camera/image_raw',
    'obstacles': '/perception/points_obstacles', 'objects': '/perception/objects',
    'tracks': '/perception/tracked_objects', 'predictions': '/perception/predicted_objects',
    'costmap': '/costmap/combined', 'odom_local': '/odometry/local', 'odom_global': '/odometry/global',
}

# rosbag topic sets
BAG_LIGHT = ['/clock', '/tf', '/tf_static', '/odom', '/imu/data', '/gps/fix', '/carla/bridge_info',
             '/carla/ground_truth/ego_odom', '/carla/ground_truth/objects', '/carla/ground_truth/boxes_3d',
             '/carla/ground_truth/boxes_2d', '/perception/objects', '/perception/tracked_objects',
             '/perception/predicted_objects', '/costmap/combined', '/costmap/stats',
             '/odometry/local', '/odometry/global', '/odometry/gps', '/cmd_vel', '/plan',
             '/gps_goal/status', '/gps_goal/route', '/carla/ego/collision', '/carla/ego/control']
BAG_FULL = BAG_LIGHT + ['/carla/lidar/points', '/carla/ground_truth/lidar_semantic',
                        '/carla/camera/image_raw', '/carla/camera/camera_info']


# ---------------------------------------------------------------- navigation geometry
def box_corners(cx, cy, yaw, x0, x1, y0, y1):
    """Corners (4, 2) of the rectangle [x0, x1] x [y0, y1] given in a frame at (cx, cy, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return local @ np.array([[c, s], [-s, c]]) + (cx, cy)


def _separated(a, b):
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return True
    return False


def _point_segment(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def rect_distance(a, b):
    """Gap between two rectangles given as corners (4, 2); 0 if they touch or overlap."""
    if not _separated(a, b):
        return 0.0
    return min(_point_segment(p, q[i], q[(i + 1) % 4])
               for p_set, q in ((a, b), (b, a)) for p in p_set for i in range(4))


# ---------------------------------------------------------------- metrics
def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def _weighted(rows, value, weight):
    num = den = 0.0
    for r in rows:
        v, w = _num(r.get(value)), _num(r.get(weight)) or 0.0
        if v is not None and w > 0:
            num, den = num + v * w, den + w
    return num / den if den else None


def extract_metrics(summaries):
    """{evaluator name: its JSON 'summary'} -> flat {metric name: number}.

    Names are '<evaluator>.<group>.<quantity>'; groups are pedestrian / vehicle, with
    _moving / _static where the evaluator splits by motion."""
    m = {}
    c = summaries.get('cloud_eval_node')
    if c:
        m['cloud.ground_precision'] = 100.0 * c['ground_precision']
        m['cloud.ground_recall'] = 100.0 * c['ground_recall']
        m['cloud.under_split_pct'] = c['under_split_pct']

    f = summaries.get('fusion_eval_node')
    if f:
        by_group = {}
        for key, r in f.items():                      # 'pedestrian 5-10 m' -> row
            by_group.setdefault(key.split()[0], []).append(r)
        for g, rows in by_group.items():
            n = sum(r['boxes'] for r in rows)
            m['fusion.%s.boxes' % g] = n
            m['fusion.%s.coverage_pct' % g] = _weighted(rows, 'coverage_pct', 'boxes')
            for r in rows:
                r['_est'] = r['boxes'] * (_num(r['coverage_pct']) or 0.0) / 100.0
            m['fusion.%s.range_mae' % g] = _weighted(rows, 'range_mae', '_est')
            m['fusion.%s.correct_pct' % g] = _weighted(rows, 'correct_pct', '_est')

    t = summaries.get('tracking_eval_node')
    if t:
        for g, r in t.get('groups', {}).items():
            m['tracking.%s.pos_mae' % g] = _num(r['pos_mae'])
            m['tracking.%s.v_moving_mae' % g] = _num(r['v_moving_mae'])
            m['tracking.%s.v_static_mae' % g] = _num(r['v_static_mae'])
            if r['actor_frames']:
                m['tracking.%s.id_switches_per_100' % g] = 100.0 * r['id_switches'] / r['actor_frames']

    p = summaries.get('prediction_eval_node')
    if p:
        for key, per_h in p.items():                  # 'pedestrian moving' -> [{h, ours, stay, in95}]
            name = key.replace(' ', '_')
            for row in per_h:
                tag = '%gs' % row['h']
                m['prediction.%s.err_%s' % (name, tag)] = _num(row['ours'])
                m['prediction.%s.in95_%s' % (name, tag)] = _num(row['in95'])
                stay = _num(row['stay'])
                if stay:
                    m['prediction.%s.gain_vs_stay_%s' % (name, tag)] = 100.0 * (1.0 - row['ours'] / stay)

    k = summaries.get('costmap_eval_node')
    if k:
        for key, (n, pct) in k.get('recall', {}).items():   # 'vehicle True' -> [actor-frames, %]
            g, moving = key.split()
            m['costmap.recall.%s_%s' % (g, 'moving' if moving == 'True' else 'static')] = _num(pct)
        m['costmap.false_obstacle_pct'] = _num(k.get('false_obstacle_pct'))
        for h, vals in k.get('dynamic', {}).items():         # '1.0' -> [n, % inside, mean cost]
            if vals and vals[0]:
                m['costmap.dynamic_%gs_pct' % float(h)] = _num(vals[1])
        m['costmap.flicker_pct'] = _num(k.get('flicker_pct'))
        m['costmap.ms_mean'] = _num(k.get('ms_mean'))
        m['costmap.ms_p95'] = _num(k.get('ms_p95'))

    loc = summaries.get('localization_eval_node')
    if loc:
        for topic, s in loc.items():                  # '/odometry/global' -> {pos_rmse_m, ...}
            if s:
                name = topic.rsplit('/', 1)[-1]
                for q in ('pos_rmse_m', 'pos_max_m', 'head_rmse_deg', 'drift_pct', 'samples', 'distance_m'):
                    m['loc.%s.%s' % (name, q)] = _num(s.get(q))
    return {k: v for k, v in m.items() if v is not None}


# ---------------------------------------------------------------- limits
OPS = {'>=': lambda a, b: a >= b, '<=': lambda a, b: a <= b, '==': lambda a, b: abs(a - b) < 1e-9}


def check(metrics, limits):
    """[(metric, op, limit)] -> rows (metric, value or None, op, limit, PASS / FAIL / MISSING)."""
    rows = []
    for name, op, limit in limits:
        v = metrics.get(name)
        status = 'MISSING' if v is None else ('PASS' if OPS[op](v, limit) else 'FAIL')
        rows.append((name, v, op, limit, status))
    return rows


def limits_for(name, stack_metrics):
    """A scenario's own limits plus the common ones its metrics can answer."""
    own = SCENARIOS[name]['limits']
    names = {n for n, _, _ in own}
    return own + [lim for lim in COMMON_LIMITS if lim[0] not in names and lim[0] in stack_metrics]


def verdict(rows):
    if any(r[4] == 'FAIL' for r in rows):
        return 'FAIL'
    if any(r[4] == 'MISSING' for r in rows):
        return 'INCOMPLETE'
    return 'PASS'


def fmt(v):
    if v is None:
        return '-'
    if isinstance(v, float) and abs(v) < 100 and v != int(v):
        return '%.2f' % v
    return '%g' % v


def format_report(results, title):
    """results: [{'name', 'about', 'verdict', 'rows', 'metrics', 'notes'}] -> markdown."""
    lines = ['# ' + title, '', '| scenario | verdict | failed / missing |', '|---|---|---|']
    for r in results:
        bad = [row[0] for row in r['rows'] if row[4] != 'PASS']
        lines.append('| %s | **%s** | %s |' % (r['name'], r['verdict'], ', '.join(bad) or '-'))
    for r in results:
        lines += ['', '## %s: %s' % (r['name'], r['verdict']), '', r['about'], '',
                  '| metric | value | limit | |', '|---|---|---|---|']
        for name, v, op, limit, status in r['rows']:
            lines.append('| %s | %s | %s %s | %s |' % (name, fmt(v), op, fmt(limit), status))
        others = sorted(k for k in r['metrics'] if k not in {row[0] for row in r['rows']})
        if others:
            lines += ['', '<details><summary>all metrics</summary>', '', '| metric | value |', '|---|---|']
            lines += ['| %s | %s |' % (k, fmt(r['metrics'][k])) for k in others]
            lines += ['', '</details>']
        for note in r.get('notes', []):
            lines += ['', '> ' + note]
    return '\n'.join(lines) + '\n'
