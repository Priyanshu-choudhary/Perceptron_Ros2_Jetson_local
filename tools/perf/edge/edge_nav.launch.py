"""Edge (Jetson) edition of nav.launch.py for profiling.

Same nodes and parameters as localization.launch.py + navigation.launch.py on
the laptop, with files referenced by path because the perceptron packages are
not built here. Headless: no RViz. Motors are never commanded (/cmd_vel of the
bridge is remapped to a sink), exactly as in the laptop harness.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetParameter
from nav2_common.launch import RewrittenYaml

W = '/work'


def generate_launch_description():
    ekf = LaunchConfiguration('ekf')
    stm32 = LaunchConfiguration('stm32')
    aruco = LaunchConfiguration('aruco')
    lean = LaunchConfiguration('lean_bridge')
    params_file = LaunchConfiguration('params_file')

    args = [
        DeclareLaunchArgument('ekf', default_value='false'),
        DeclareLaunchArgument('stm32', default_value='true'),
        DeclareLaunchArgument('aruco', default_value='true'),
        DeclareLaunchArgument('lean_bridge', default_value='false'),
        DeclareLaunchArgument('bond', default_value='true'),
        DeclareLaunchArgument('use_composition', default_value='False'),
        DeclareLaunchArgument('nav_profile', default_value='dwb'),
        DeclareLaunchArgument('params_file', default_value=f'{W}/params/base.yaml'),
    ]

    bridge_publish_tf = PythonExpression(["not ('", ekf, "'.lower() in ('true', '1'))"])
    urdf = open(f'{W}/robot.urdf').read()

    no_bond = SetParameter(name='bond_timeout', value=0.0,
                           condition=UnlessCondition(LaunchConfiguration('bond')))

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               name='robot_state_publisher', output='screen',
               parameters=[{'robot_description': urdf, 'use_sim_time': False}])
    jsp = Node(package='joint_state_publisher', executable='joint_state_publisher',
               name='joint_state_publisher', output='screen',
               parameters=[{'use_sim_time': False}])

    bridge_params = [f'{W}/config/gyro_params.yaml', {
        'jetson_ip': '127.0.0.1', 'telemetry_port': 5555, 'cmd_port': 5556,
        'laser_frame_id': 'laser_link', 'base_frame_id': 'base_footprint',
        'odom_frame_id': 'odom', 'imu_frame_id': 'imu_link',
        'publish_tf': bridge_publish_tf, 'auto_arm': True, 'use_sim_time': False}]
    bridge_remap = [('/cmd_vel', '/cmd_vel_profiler_sink')]
    bridge_stock = Node(executable='python3', arguments=['-m', 'perceptron_hardware.jetson_bridge_node'],
                        name='jetson_bridge_node', output='screen', condition=UnlessCondition(lean),
                        parameters=bridge_params, remappings=bridge_remap)
    bridge_lean = Node(executable='python3', arguments=[f'{W}/lean/jetson_bridge_node_lean.py'],
                       name='jetson_bridge_node', output='screen', condition=IfCondition(lean),
                       parameters=bridge_params, remappings=bridge_remap)

    battery = Node(executable='python3', arguments=['-m', 'perceptron_hardware.battery_node'],
                   name='battery_node', output='screen', condition=IfCondition(stm32),
                   parameters=[f'{W}/config/battery_real.yaml'])
    ekf_node = Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
                    output='screen', condition=IfCondition(ekf),
                    parameters=[f'{W}/config/ekf_real.yaml'])
    map_server = Node(package='nav2_map_server', executable='map_server', name='map_server',
                      output='screen',
                      parameters=[{'yaml_filename': f'{W}/maps/room_map.yaml', 'use_sim_time': False,
                                   'topic_name': 'map', 'frame_id': 'map'}])
    # AMCL reads the PACKAGE nav2_params.yaml on the laptop (not params_file)
    amcl = Node(package='nav2_amcl', executable='amcl', name='amcl', output='screen',
                parameters=[f'{W}/params/nav2_params_pkg.yaml',
                            {'use_sim_time': False, 'set_initial_pose': False}])
    lcm_loc = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                   name='lifecycle_manager_localization', output='screen',
                   parameters=[{'use_sim_time': False, 'autostart': True,
                                'node_names': ['map_server', 'amcl'],
                                'bond_timeout': PythonExpression(
                                    ["0.0 if '", LaunchConfiguration('bond'), "'.lower() == 'false' else 4.0"])}])
    aruco_node = Node(executable='python3', arguments=['-m', 'perceptron_navigation.aruco_localizer_node'],
                      name='aruco_localizer_node', output='screen', condition=IfCondition(aruco),
                      parameters=[f'{W}/config/aruco_localization.yaml',
                                  {'use_sim_time': False, 'jetson_ip': '127.0.0.1',
                                   'marker_map_path': f'{W}/config/marker_map.yaml'}])

    bt_xml = PythonExpression(["'" + W + "/behavior_trees/nav_to_pose_' + '",
                               LaunchConfiguration('nav_profile'), "' + '.xml'"])
    profile_params = RewrittenYaml(
        source_file=params_file, root_key='',
        param_rewrites={'default_nav_to_pose_bt_xml': bt_xml,
                        'initial_pose.x': '0.0', 'initial_pose.y': '0.0', 'initial_pose.yaw': '0.0'},
        convert_types=True)
    container = Node(condition=IfCondition(LaunchConfiguration('use_composition')),
                     package='rclcpp_components', executable='component_container_isolated',
                     name='nav2_container', output='screen',
                     parameters=[{'use_sim_time': False, 'autostart': True}],
                     remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')])
    navigation_inc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(f'{W}/navigation_launch.py'),
        launch_arguments={'use_sim_time': 'false', 'params_file': profile_params,
                          'autostart': 'true',
                          'use_composition': LaunchConfiguration('use_composition')}.items())
    # scoped: a global SetParameter would add -p to AMCL and flip YAML/dict precedence
    navigation = GroupAction(actions=[no_bond, navigation_inc])

    return LaunchDescription(args + [
        LogInfo(msg='PROFILER launch (edge)'), container,
        rsp, jsp, bridge_stock, bridge_lean, battery, ekf_node, map_server, amcl, lcm_loc,
        aruco_node, navigation])
