"""Everything the laptop ran, on the Jetson Nano, headless and optimised.

    ros2 launch perceptron_edge edge.launch.py mode:=slam          # build a map (and navigate on it)
    ros2 launch perceptron_edge edge.launch.py mode:=nav map:=room_map

Normally started through the `robot` command on the Jetson (host/robot), which
also manages the serial bridge (jetson_robot_bridge.py) and the ROS 2 chroot.

MODES

    slam   slam_toolbox builds the map and owns map -> odom. With nav:=true
           (default) Nav2 also runs on the growing map, so goals work while
           mapping. Save with `robot save-map NAME`.
    nav    map_server + AMCL on a saved map own map -> odom; Nav2 on top.
           Seed AMCL with Foxglove's "Publish pose estimate", `robot pose`, or
           aruco:=true.

SWITCHES (all optional)

    map:=NAME|PATH   nav mode map: a name in the maps folder or a .yaml path
    ekf:=true        robot_localization EKF owns odom -> base_footprint
    aruco:=true      start aruco_localizer_node (nav mode). Needs the camera and
                     the serial bridge running WITHOUT --no-aruco; `robot start`
                     handles the bridge. Off by default: nothing runs for it.
    nav:=false       slam mode without Nav2 (drive with teleop)
    teleop:=true     CT6B RC receiver on this computer, teleop_port:=/dev/ttyUSBx
                     (explicit port - auto-detect could open the LiDAR/STM32)
    motors:=false    bench mode: Nav2 runs, the motors never receive /cmd_vel
    foxglove:=false  no laptop view
    log_level:=info  Nav2 / SLAM verbosity (default warn)

WHAT IS DIFFERENT FROM THE LAPTOP BRINGUP, AND WHY (measured, tools/perf)

    no RViz, no ArUco node unless asked       RViz 67% of a core, ArUco 59%
    lifecycle bonds off (bond_timeout: 0)     10 Hz heartbeats x 9 on one shared
                                              topic; every server paid for it
    bridge: odom/TF 50 Hz, battery 2 Hz,      was ~100 Hz each for consumers that
    IMU only when subscribed, no raw replies  need 30 Hz or less
    bridge: stm32_time_correction             odom/IMU stamped with the STM32's
                                              10 ms clock, not the Jetson's
                                              ~85 ms serial-read bursts
    nav2_edge.yaml / slam_edge.yaml           see those files
    smoother_server not started               the DWB behaviour tree never calls it
    joint_state_publisher at 2 Hz             wheel joints are cosmetic (always 0)
    DDS on 127.0.0.1, UDP only (`robot`)      DDS stays on the robot; the laptop
                                              views through Foxglove instead. No
                                              shared memory: it got stuck at 3.8
                                              cores (fastdds_localhost_udp.xml)

Crash handling: with bonds off the lifecycle manager no longer notices a dead
Nav2 server, so the non-lifecycle nodes (bridge, battery, state publishers,
Foxglove) respawn, and `robot status` shows any Nav2 server that has died.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml

WS_ROOT = os.environ.get('PERCEPTRON_WS', '/opt/perceptron_ws')
MAP_DIR = os.environ.get('PERCEPTRON_MAPS', os.path.join(WS_ROOT, 'maps'))
MARKER_MAP = os.environ.get('PERCEPTRON_MARKER_MAP', os.path.join(WS_ROOT, 'config', 'marker_map.yaml'))

NAV2_NODES = ['controller_server', 'planner_server', 'behavior_server',
              'bt_navigator', 'waypoint_follower', 'velocity_smoother']


def _true(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in ('true', '1', 'yes', 'on')


def _resolve_map(value):
    if value.endswith('.yaml') and os.path.isfile(value):
        return value
    for cand in (os.path.join(MAP_DIR, value), os.path.join(MAP_DIR, value + '.yaml'),
                 os.path.join(get_package_share_directory('perceptron_navigation'), 'maps', value + '.yaml')):
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(f'map "{value}" not found (looked in {MAP_DIR} and perceptron_navigation/maps)')


def _setup(context):
    mode = LaunchConfiguration('mode').perform(context).strip().lower()
    if mode not in ('slam', 'nav'):
        raise RuntimeError(f'mode must be slam or nav, not "{mode}"')
    ekf = _true(context, 'ekf')
    aruco = _true(context, 'aruco')
    nav = _true(context, 'nav') or mode == 'nav'
    motors = _true(context, 'motors')
    foxglove = _true(context, 'foxglove')
    teleop = _true(context, 'teleop')
    log_level = LaunchConfiguration('log_level').perform(context)
    ros_log = ['--ros-args', '--log-level', log_level]

    pkg_nav = get_package_share_directory('perceptron_navigation')
    pkg_hw = get_package_share_directory('perceptron_hardware')
    pkg_desc = get_package_share_directory('perceptron_robot_description')
    pkg_bringup = get_package_share_directory('perceptron_robot_bringup')
    pkg_edge = get_package_share_directory('perceptron_edge')

    actions = [LogInfo(msg=f'perceptron_edge: mode={mode} nav={nav} ekf={ekf} aruco={aruco} '
                           f'motors={motors} foxglove={foxglove} teleop={teleop}')]

    # ---------------------------------------------------------------- robot
    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(pkg_desc, 'urdf', 'perceptron_robot.xacro'), ' is_sim:=false']),
        value_type=str)
    actions += [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='robot_state_publisher', output='screen', respawn=True, respawn_delay=2.0,
             parameters=[{'robot_description': robot_description, 'use_sim_time': False}]),
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             name='joint_state_publisher', output='screen', respawn=True, respawn_delay=2.0,
             parameters=[{'use_sim_time': False, 'rate': 2}]),
        Node(package='perceptron_hardware', executable='jetson_bridge_node',
             name='jetson_bridge_node', output='screen', respawn=True, respawn_delay=2.0,
             parameters=[os.path.join(pkg_hw, 'config', 'gyro_params.yaml'), {
                 'jetson_ip': '127.0.0.1', 'telemetry_port': 5555, 'cmd_port': 5556,
                 'laser_frame_id': 'laser_link', 'base_frame_id': 'base_footprint',
                 'odom_frame_id': 'odom', 'imu_frame_id': 'imu_link',
                 # exactly one publisher of odom -> base_footprint
                 'publish_tf': not ekf,
                 'auto_arm': True, 'use_sim_time': False,
                 'odom_publish_hz': 50.0, 'battery_publish_hz': 2.0, 'imu_lazy': True,
                 # true STM32 sample times instead of the Jetson's bursty parse times
                 'stm32_time_correction': True,
                 # odom TF 150 ms ahead (predicted): telemetry arrives in ~85 ms bursts
                 'tf_future_dating': 0.15}],
             remappings=[] if motors else [('/cmd_vel', '/cmd_vel_motors_disabled')]),
        Node(package='perceptron_hardware', executable='battery_node', name='battery_node',
             output='screen', respawn=True, respawn_delay=2.0,
             parameters=[os.path.join(pkg_hw, 'config', 'battery_real.yaml')]),
    ]
    if ekf:
        actions.append(Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
                            output='screen', arguments=ros_log,
                            parameters=[os.path.join(pkg_bringup, 'config', 'ekf_real.yaml')]))

    # -------------------------------------------------------- localisation
    nav2_base = os.path.join(pkg_nav, 'config', 'nav2_params.yaml')
    nav2_edge = RewrittenYaml(
        source_file=os.path.join(pkg_edge, 'config', 'nav2_edge.yaml'), root_key='',
        # nav_to_pose_dwb.xml plus "back off the charger first" (dock_guard)
        param_rewrites={'default_nav_to_pose_bt_xml':
                        os.path.join(pkg_edge, 'behavior_trees', 'navigate_to_pose.xml')},
        convert_types=True)

    if mode == 'slam':
        actions.append(Node(
            package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
            output='screen', arguments=ros_log,
            parameters=[os.path.join(pkg_nav, 'config', 'slam_toolbox.yaml'),
                        os.path.join(pkg_edge, 'config', 'slam_edge.yaml')]))
    else:
        map_yaml = _resolve_map(LaunchConfiguration('map').perform(context))
        actions += [
            LogInfo(msg=f'perceptron_edge: map {map_yaml}'),
            Node(package='nav2_map_server', executable='map_server', name='map_server',
                 output='screen', arguments=ros_log,
                 parameters=[{'yaml_filename': map_yaml, 'use_sim_time': False,
                              'topic_name': 'map', 'frame_id': 'map'}]),
            Node(package='nav2_amcl', executable='amcl', name='amcl', output='screen',
                 arguments=ros_log, parameters=[nav2_base, nav2_edge]),
            # Lifecycle managers stay at INFO: they only log at startup/shutdown,
            # and `robot start` waits for their "Managed nodes are active".
            Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                 name='lifecycle_manager_localization', output='screen',
                 parameters=[{'use_sim_time': False, 'autostart': True,
                              'node_names': ['map_server', 'amcl'], 'bond_timeout': 0.0}]),
        ]
        if aruco:
            actions.append(Node(
                package='perceptron_navigation', executable='aruco_localizer_node',
                name='aruco_localizer_node', output='screen',
                parameters=[os.path.join(pkg_nav, 'config', 'aruco_localization.yaml'),
                            {'use_sim_time': False, 'jetson_ip': '127.0.0.1',
                             'marker_map_path': MARKER_MAP}]))

    # ---------------------------------------------------------------- Nav2
    if nav:
        # Before bt_navigator: its tree calls /dock/undock_if_docked, and a BT
        # service node whose server is missing fails the tree at load time.
        # INFO whatever log_level says: it is silent until it backs off the
        # charger, and that is worth seeing in `robot logs`.
        actions.append(Node(package='perceptron_edge', executable='dock_guard', name='dock_guard',
                            output='screen', respawn=True, respawn_delay=2.0,
                            arguments=['--ros-args', '--log-level', 'info']))
        remap_ctrl = [('cmd_vel', 'cmd_vel_nav')]
        remap_smoother = [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]
        for name, pkg in (('controller_server', 'nav2_controller'), ('planner_server', 'nav2_planner'),
                          ('behavior_server', 'nav2_behaviors'), ('bt_navigator', 'nav2_bt_navigator'),
                          ('waypoint_follower', 'nav2_waypoint_follower'),
                          ('velocity_smoother', 'nav2_velocity_smoother')):
            actions.append(Node(
                package=pkg, executable=name, name=name, output='screen', arguments=ros_log,
                parameters=[nav2_base, nav2_edge],
                remappings=(remap_ctrl if name == 'controller_server'
                            else remap_smoother if name == 'velocity_smoother' else [])))
        actions.append(Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_navigation', output='screen',
            parameters=[{'use_sim_time': False, 'autostart': True,
                         'node_names': NAV2_NODES, 'bond_timeout': 0.0}]))

    # -------------------------------------------------------------- extras
    if foxglove:
        actions += [
            Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
                 output='screen', respawn=True, respawn_delay=2.0, arguments=ros_log,
                 parameters=[os.path.join(pkg_edge, 'config', 'foxglove.yaml')]),
            # 5 Hz copies of /scan and /odom for viewing; each subscribes to its
            # source only while a Foxglove client is displaying it.
            Node(package='topic_tools', executable='throttle', name='viz_scan_throttle',
                 output='screen', respawn=True, respawn_delay=2.0,
                 arguments=['messages', '/scan', '5.0', '/viz/scan'],
                 parameters=[{'lazy': True, 'use_sim_time': False}]),
            Node(package='topic_tools', executable='throttle', name='viz_odom_throttle',
                 output='screen', respawn=True, respawn_delay=2.0,
                 arguments=['messages', '/odom', '5.0', '/viz/odom'],
                 parameters=[{'lazy': True, 'use_sim_time': False}]),
        ]
    if teleop:
        port = LaunchConfiguration('teleop_port').perform(context).strip()
        if not port:
            raise RuntimeError('teleop:=true needs teleop_port:=/dev/ttyUSBx (never auto-detect: '
                               'ttyUSB0/1 are the LiDAR and the STM32)')
        actions.append(Node(
            package='ct6b_teleop', executable='ct6b_teleop_node', name='ct6b_teleop_node',
            output='screen', respawn=True, respawn_delay=2.0,
            parameters=[os.path.join(get_package_share_directory('ct6b_teleop'), 'config', 'ct6b_params.yaml'),
                        {'serial_port': port, 'auto_detect_port': False}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='nav', description='slam | nav'),
        DeclareLaunchArgument('map', default_value='room_map', description='nav mode: map name or .yaml path'),
        DeclareLaunchArgument('ekf', default_value='false'),
        DeclareLaunchArgument('aruco', default_value='false'),
        DeclareLaunchArgument('nav', default_value='true', description='slam mode: also run Nav2'),
        DeclareLaunchArgument('motors', default_value='true', description='false = bench mode, motors ignored'),
        DeclareLaunchArgument('foxglove', default_value='true'),
        DeclareLaunchArgument('teleop', default_value='false'),
        DeclareLaunchArgument('teleop_port', default_value=''),
        DeclareLaunchArgument('log_level', default_value='warn'),
        OpaqueFunction(function=_setup),
    ])
