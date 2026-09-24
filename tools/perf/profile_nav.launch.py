"""Profiling copy of perceptron_robot_bringup/nav.launch.py.

Identical node set and argument passing to the installed nav.launch.py, with
three harness-only additions:

  params_file  Nav2 parameter file handed to navigation.launch.py (so a tuned
               variant can be measured without editing the package).
  aruco        passed to localization.launch.py (nav.launch.py never exposes
               it, so the stock command always starts aruco_localizer_node).
  disarm       true (default): /cmd_vel is remapped to /cmd_vel_profiler_sink
               for the localization include ONLY, i.e. for jetson_bridge_node.
               Nav2 still computes and publishes /cmd_vel; the motors never
               receive it.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, LogInfo)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetParameter, SetRemap


def generate_launch_description():
    pkg_bringup = get_package_share_directory('perceptron_robot_bringup')
    pkg_nav = get_package_share_directory('perceptron_navigation')

    map_yaml = LaunchConfiguration('map')
    use_jetson = LaunchConfiguration('use_jetson')
    jetson = LaunchConfiguration('jetson')
    jetson_ip = LaunchConfiguration('jetson_ip')
    resolved_jetson_ip = PythonExpression([
        "'", jetson, "'.strip() if '", jetson, "'.strip() != '' else '", jetson_ip, "'.strip()"
    ])

    args = [
        DeclareLaunchArgument('map', default_value=os.path.join(pkg_nav, 'maps', 'room2_map.yaml')),
        DeclareLaunchArgument('use_jetson', default_value='true'),
        DeclareLaunchArgument('jetson', default_value=''),
        DeclareLaunchArgument('jetson_ip', default_value=os.environ.get('JETSON_IP', '192.168.1.7')),
        DeclareLaunchArgument('lidar_port', default_value=''),
        DeclareLaunchArgument('stm32_port', default_value=''),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('ekf', default_value='true'),
        DeclareLaunchArgument('stm32', default_value='true'),
        DeclareLaunchArgument('nav_profile', default_value='dwb'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('overlay', default_value='false'),
        DeclareLaunchArgument('overlay_jetson_ip', default_value='192.168.1.7'),
        # harness-only
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(pkg_nav, 'config', 'nav2_params.yaml')),
        DeclareLaunchArgument('aruco', default_value='true'),
        DeclareLaunchArgument('disarm', default_value='true'),
        # nav2_bringup/navigation_launch.py reads this by config inheritance.
        DeclareLaunchArgument('use_composition', default_value='False'),
        # false: lifecycle managers skip bond creation (bond_timeout 0.0)
        DeclareLaunchArgument('bond', default_value='true'),
        # true: localization include uses the patched lean jetson_bridge_node
        DeclareLaunchArgument('lean_bridge', default_value='false'),
    ]

    # Scoped to the Nav2 include ONLY. A global SetParameter adds `-p` to every
    # node, and any `-p` makes a node's YAML beat its inline dict: AMCL would
    # silently get use_sim_time/set_initial_pose from nav2_params.yaml. The
    # localization lifecycle manager (2 bonds) therefore keeps its bonds.
    no_bond = SetParameter(name='bond_timeout', value=0.0,
                           condition=UnlessCondition(LaunchConfiguration('bond')))

    # Only used with use_composition:=True - navigation_launch.py loads every
    # Nav2 server into this one process instead of starting seven.
    nav2_container = Node(
        condition=IfCondition(LaunchConfiguration('use_composition')),
        package='rclcpp_components', executable='component_container_isolated',
        name='nav2_container', output='screen',
        parameters=[{'use_sim_time': False, 'autostart': True}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')])

    loc_args = {
        'map': map_yaml,
        'use_jetson': use_jetson,
        'jetson': resolved_jetson_ip,
        'jetson_ip': resolved_jetson_ip,
        'lidar_port': LaunchConfiguration('lidar_port'),
        'stm32_port': LaunchConfiguration('stm32_port'),
        'rviz': LaunchConfiguration('rviz'),
        'ekf': LaunchConfiguration('ekf'),
        'stm32': LaunchConfiguration('stm32'),
        'aruco': LaunchConfiguration('aruco'),
    }.items()
    loc_src = PythonLaunchDescriptionSource(PythonExpression([
        "'" + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lean', 'localization_lean_bridge.launch.py')
        + "' if '", LaunchConfiguration('lean_bridge'), "'.lower() == 'true' else '"
        + os.path.join(pkg_bringup, 'launch', 'localization.launch.py') + "'"]))

    robot_and_localization_disarmed = GroupAction(
        condition=IfCondition(LaunchConfiguration('disarm')),
        actions=[SetRemap(src='/cmd_vel', dst='/cmd_vel_profiler_sink'),
                 IncludeLaunchDescription(loc_src, launch_arguments=loc_args)])
    robot_and_localization_armed = GroupAction(
        condition=UnlessCondition(LaunchConfiguration('disarm')),
        actions=[IncludeLaunchDescription(loc_src, launch_arguments=loc_args)])

    navigation_inc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav, 'launch', 'navigation.launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'localization': 'external',
            'slam': 'false',
            'nav_profile': LaunchConfiguration('nav_profile'),
            'autostart': LaunchConfiguration('autostart'),
            'relay': 'false',
            'robot_cmd_vel_topic': '/cmd_vel',
            'nav_cmd_vel_topic': '/cmd_vel',
            'overlay': LaunchConfiguration('overlay'),
            'overlay_jetson_ip': resolved_jetson_ip,
            'params_file': LaunchConfiguration('params_file'),
        }.items(),
    )
    navigation = GroupAction(actions=[no_bond, navigation_inc])

    return LaunchDescription(args + [
        LogInfo(msg='PROFILER launch'),
        nav2_container,
        robot_and_localization_disarmed,
        robot_and_localization_armed,
        navigation,
    ])
