#!/usr/bin/env python3
"""Write Nav2 parameter variants derived from the installed nav2_params.yaml.

Every variant (including 'base') carries ONE harness-only change:
controller_server.odom_topic -> /profiler/controller_odom, the velocity
loop-back that lets DWB/MPPI sample at the speed they commanded while the
robot is physically parked. Nothing else in 'base' differs from the package.
"""
import copy
import os
import sys

import yaml

SRC = os.path.expanduser('~/perceptron_test_ws/src/perceptron_navigation/config/nav2_params.yaml')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params')


def P(d, *path):
    for k in path:
        d = d[k]
    return d


def cs(d):
    return P(d, 'controller_server', 'ros__parameters')


def fp(d):
    return cs(d)['FollowPath']


def lcm(d):
    return P(d, 'local_costmap', 'local_costmap', 'ros__parameters')


def gcm(d):
    return P(d, 'global_costmap', 'global_costmap', 'ros__parameters')


def base(d):
    cs(d)['odom_topic'] = '/profiler/controller_odom'


def dwb_nodebug(d):
    f = fp(d)
    f['debug_trajectory_details'] = False
    f['publish_evaluation'] = False       # records all 800 trajectories each cycle when true
    f['publish_trajectories'] = False     # ditto (MarkerArray)
    f['publish_global_plan'] = False      # /received_global_plan (debug copy)
    f['publish_transformed_plan'] = False  # /transformed_global_plan (debug)
    f['publish_cost_grid_pc'] = False
    # publish_local_plan stays true: RViz "Local Plan" display uses it


def ctrl_freq10(d):
    # MPPI refuses to configure when the controller period exceeds its model_dt
    # (0.05 s), which aborts the whole Nav2 bringup, so it must go with it.
    cs(d)['controller_frequency'] = 10.0
    cs(d)['controller_plugins'] = ['FollowPath']
    cs(d).pop('FollowPathMPPI', None)


def ctrl_samples(d):
    f = fp(d)
    f['vx_samples'] = 10
    f['vtheta_samples'] = 20


def ctrl_gran(d):
    f = fp(d)
    f['angular_granularity'] = 0.05
    f['linear_granularity'] = 0.1


def ctrl_simtime(d):
    fp(d)['sim_time'] = 1.2


def ctrl_reduced(d):
    ctrl_freq10(d)
    ctrl_samples(d)
    ctrl_gran(d)


def gcm_static(d):
    # sized to the saved map (241 x 219 cells) instead of a 48 m rolling window
    g = gcm(d)
    g['rolling_window'] = False
    g.pop('width', None)
    g.pop('height', None)


def gcm_res10(d):
    gcm(d)['resolution'] = 0.10


def gcm_window16(d):
    g = gcm(d)
    g['width'] = 16
    g['height'] = 16


def cm_pub_lean(d):
    for g in (gcm(d), lcm(d)):
        g['always_send_full_costmap'] = False
    gcm(d)['publish_frequency'] = 0.5


def cm_no_terrain(d):
    # /traversability_obstacles has no publisher on the indoor robot
    for g in (gcm(d), lcm(d)):
        g['obstacle_layer']['observation_sources'] = 'scan'
        g['obstacle_layer'].pop('terrain', None)


def lcm_rate(d):
    lcm(d)['update_frequency'] = 2.0
    gcm(d)['update_frequency'] = 0.5


def cm_lean(d):
    gcm_static(d)
    cm_pub_lean(d)
    cm_no_terrain(d)


def plugins_lean(d):
    c = cs(d)
    c['controller_plugins'] = ['FollowPath']
    c.pop('FollowPathMPPI', None)
    p = P(d, 'planner_server', 'ros__parameters')
    p['planner_plugins'] = ['Smac2D']
    for k in ('GridBased', 'GridBasedUnknown', 'ThetaStar'):
        p.pop(k, None)
    b = P(d, 'behavior_server', 'ros__parameters')
    b['behavior_plugins'] = ['spin', 'backup', 'wait']
    b.pop('drive_on_heading', None)
    b.pop('assisted_teleop', None)


def optimized(d):
    dwb_nodebug(d)
    ctrl_reduced(d)
    cm_lean(d)
    plugins_lean(d)


def optimized_keep_ctrl(d):
    """Everything except the controller retune (behaviour-neutral changes only)."""
    dwb_nodebug(d)
    cm_lean(d)
    plugins_lean(d)


VARIANTS = {k: v for k, v in globals().items()
            if callable(v) and k not in ('P', 'cs', 'fp', 'lcm', 'gcm', 'main')}


def main():
    os.makedirs(OUT, exist_ok=True)
    src = yaml.safe_load(open(SRC))
    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        d = copy.deepcopy(src)
        base(d)
        if name != 'base':
            for part in name.split('+'):
                VARIANTS[part](d)
        path = os.path.join(OUT, f'{name}.yaml')
        with open(path, 'w') as f:
            yaml.safe_dump(d, f, sort_keys=False)
        print(os.path.abspath(path))


if __name__ == '__main__':
    main()
