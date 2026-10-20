"""The rover's own localization stack on simulated sensors, plus its evaluator (Stage 3).

    # Windows: CarlaUE4.exe.  WSL terminal 1 (start this first; it waits for /clock):
    ros2 launch perceptron_carla sim_localization.launch.py                 # local + global EKF
    ros2 launch perceptron_carla sim_localization.launch.py global:=false   # local EKF only
    # WSL terminal 2 -- bridge leaves odom/map TF to the EKFs and holds the car still 10 s:
    ros2 run perceptron_carla carla_bridge --autopilot --tf-mode ekf

Same nodes, names and remaps as perceptron_edge's GPS mode, and the same
dual_ekf_navsat.yaml; only sim_ekf_overrides.yaml is added on top. With
global:=false nothing publishes map -> odom, so a static identity stands in.

fixes:=false (default) reproduces perceptron_edge's GPS mode exactly, bugs included.
fixes:=true applies the three launch/config fixes found in Stage 3 (README, "Stage 3
findings"); none of them touches the real robot's files:
  1. navsat_transform's IMU input is 'imu' in ROS 2, so remap 'imu' -> /imu/data
     (edge.launch.py's ('/imu/data', '/imu/data') is a no-op; navsat never initialises)
  2. navsat takes /odometry/global, not /odometry/local, as robot_localization's
     dual-EKF example does (otherwise the GPS odometry is in odom and the global EKF
     moves it with its own map -> odom: a feedback loop)
  3. the local EKF stops fusing x/y from /odom, which stm32_bridge_node always
     publishes as (0, 0) with variance 1e6
The fourth finding, vy variance 1e6 on /odom, is the driver's message, so it is
the bridge's --odom-vy-cov, not a launch argument.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

# dual_ekf_navsat.yaml's local odom0_config with x and y switched off
ODOM0_NO_POSE = [False, False, False, False, False, False,
                 True, True, False, False, False, True,
                 False, False, False]


def _nodes(context):
    def arg(name):
        return context.launch_configurations[name]

    real = os.path.join(get_package_share_directory('perceptron_navigation'),
                        'config', 'dual_ekf_navsat.yaml')
    sim = os.path.join(get_package_share_directory('perceptron_carla'),
                       'config', 'sim_ekf_overrides.yaml')
    params = [real, sim]
    fixes = arg('fixes').lower() == 'true'
    local_params = list(params)
    if fixes:
        local_params.append({'odom0_config': ODOM0_NO_POSE})
    sim_time = {'use_sim_time': True}
    use_global = arg('global').lower() == 'true'
    if fixes:
        navsat_remaps = [('imu', '/imu/data'), ('gps/fix', '/gps/fix'),
                         ('odometry/filtered', '/odometry/global'), ('odometry/gps', '/odometry/gps')]
    else:   # exactly perceptron_edge/launch/edge.launch.py
        navsat_remaps = [('/imu/data', '/imu/data'), ('/gps/fix', '/gps/fix'),
                         ('/odometry/filtered', '/odometry/local'), ('/odometry/gps', '/odometry/gps')]

    nodes = [Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node_local',
                  output='screen', parameters=local_params,
                  remappings=[('/odometry/filtered', '/odometry/local')])]
    if use_global:
        nodes += [
            Node(package='robot_localization', executable='navsat_transform_node',
                 name='navsat_transform_node', output='screen', parameters=params,
                 remappings=navsat_remaps),
            Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node_global',
                 output='screen', parameters=params,
                 remappings=[('/odometry/filtered', '/odometry/global')]),
        ]
    else:
        nodes.append(Node(package='tf2_ros', executable='static_transform_publisher',
                          name='map_to_odom_identity', parameters=[sim_time],
                          arguments=['--frame-id', 'map', '--child-frame-id', 'odom']))
    if arg('eval').lower() == 'true':
        nodes.append(Node(package='perceptron_carla', executable='localization_eval_node',
                          name='localization_eval_node', output='screen',
                          parameters=[sim_time, {'report_period': float(arg('report_period'))}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('global', default_value='true',
                              description='navsat_transform + global EKF (map -> odom)'),
        DeclareLaunchArgument('eval', default_value='true', description='run localization_eval_node'),
        DeclareLaunchArgument('report_period', default_value='30.0',
                              description='evaluator report period, sim seconds'),
        DeclareLaunchArgument('fixes', default_value='false',
                              description='true applies the three Stage 3 launch/config fixes'),
        OpaqueFunction(function=_nodes),
    ])
