"""The real robot localizing against a SAVED map with AMCL, instead of SLAM.

    ros2 launch perceptron_robot_bringup localization.launch.py \
        map:=$HOME/perceptron_test_ws/maps/room.yaml

Same hardware bringup as robot.launch.py -- lidar, ECU, EKF, RViz -- but the
map comes from disk and AMCL localizes in it rather than slam_toolbox building
a new one. Use this once the room is mapped: the map stops changing, startup is
quicker, and the CPU that slam_toolbox spent on scan matching goes back.

SLAM vs AMCL, in terms of who owns which transform:

    slam_toolbox   builds the map AND publishes map -> odom
    AMCL           reads a fixed map AND publishes map -> odom

Either way exactly one of them may run. Both at once means two publishers on
map -> odom and TF interleaves them, which looks like the robot teleporting
between two poses several times a second.

AMCL starts with no idea where the robot is. Give it a starting pose with the
"2D Pose Estimate" button in RViz, or it will happily localize you into the
wrong room and stay confident about it.

WHERE THE SENSORS COME FROM

Like robot.launch.py, this defaults to the Jetson ZeroMQ bridge: one node
(jetson_bridge_node) receives /scan, /odom and /imu/data_raw over Wi-Fi from
jetson_robot_bridge.py running on the Nano. The lidar and STM32 are plugged
into the JETSON, not into this laptop, so the direct-USB nodes must not run --
they would sit forever retrying /dev/ttyUSB0, which does not exist here.

Pass use_jetson:=false only when the cables are genuinely in this machine.

Arguments:
    map          path to the .yaml written by map_saver_cli.
                 Default: the packaged room_map.yaml.
    use_jetson   receive sensors over ZeroMQ from the Nano. Default true.
    jetson_ip    address of the Nano. Default 192.168.1.7 .
    rviz         open RViz. Default true.
    ekf/stm32    as robot.launch.py
    lidar_port   override port detection (use_jetson:=false only)
    stm32_port   override port detection (use_jetson:=false only)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def _resolve_ports(context):
    # Nothing to detect when the sensors are on the Jetson: the ports being
    # looked for are on the other machine, and scanning for them here only
    # produces a misleading "detected lidar on NOTHING".
    if LaunchConfiguration('use_jetson').perform(context).lower() in ('true', '1'):
        return []

    lidar = LaunchConfiguration('lidar_port').perform(context)
    stm32 = LaunchConfiguration('stm32_port').perform(context)
    if lidar and stm32:
        return []
    try:
        from perceptron_hardware.port_detect import detect_ports
        found = detect_ports()
    except Exception as exc:
        return [LogInfo(msg='port detection failed (%s); falling back' % exc)]
    msgs = []
    if not lidar:
        lidar = found.get('lidar', '')
        msgs.append(LogInfo(msg='detected lidar on %s' % (lidar or 'NOTHING')))
    if not stm32:
        stm32 = found.get('stm32', '')
        msgs.append(LogInfo(msg='detected stm32 on %s' % (stm32 or 'NOTHING')))
    context.launch_configurations['lidar_port'] = lidar or '/dev/ttyUSB0'
    context.launch_configurations['stm32_port'] = stm32 or '/dev/ttyUSB1'
    return msgs


def generate_launch_description():
    pkg_desc = get_package_share_directory('perceptron_robot_description')
    pkg_hw = get_package_share_directory('perceptron_hardware')
    pkg_bringup = get_package_share_directory('perceptron_robot_bringup')
    pkg_nav = get_package_share_directory('perceptron_navigation')

    xacro_file = os.path.join(pkg_desc, 'urdf', 'perceptron_robot.xacro')

    map_yaml = LaunchConfiguration('map')
    use_jetson = LaunchConfiguration('use_jetson')
    jetson = LaunchConfiguration('jetson')
    jetson_ip = LaunchConfiguration('jetson_ip')
    resolved_jetson_ip = PythonExpression([
        "'", jetson, "'.strip() if '", jetson, "'.strip() != '' else '", jetson_ip, "'.strip()"
    ])
    lidar_port = LaunchConfiguration('lidar_port')
    stm32_port = LaunchConfiguration('stm32_port')
    rviz = LaunchConfiguration('rviz')
    ekf = LaunchConfiguration('ekf')
    stm32 = LaunchConfiguration('stm32')
    aruco = LaunchConfiguration('aruco')
    marker_map = LaunchConfiguration('marker_map')

    args = [
        DeclareLaunchArgument(
            'map', default_value=os.path.join(pkg_nav, 'maps', 'room_map.yaml'),
            description='path to map .yaml'),
        DeclareLaunchArgument('use_jetson', default_value='true',
                              description='sensors arrive over ZeroMQ from the Nano'),
        DeclareLaunchArgument('jetson', default_value='',
                              description='Jetson Nano IP address (e.g. jetson:=192.168.1.7)'),
        DeclareLaunchArgument('jetson_ip',
                              default_value=os.environ.get('JETSON_IP', '192.168.1.7'),
                              description='Jetson Nano IP address (alias for jetson)'),
        DeclareLaunchArgument('lidar_port', default_value=''),
        DeclareLaunchArgument('stm32_port', default_value=''),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('ekf', default_value='true'),
        DeclareLaunchArgument('stm32', default_value='true'),
        DeclareLaunchArgument(
            'aruco', default_value='true',
            description='seed AMCL from a surveyed wall board instead of a '
                        'hand-placed 2D Pose Estimate'),
        DeclareLaunchArgument(
            'marker_map',
            default_value=os.path.expanduser(
                '~/perceptron_test_ws/config/marker_map.yaml'),
            description='surveyed board poses, written by '
                        'marker_localization.launch.py mode:=teach'),
        OpaqueFunction(function=_resolve_ports),
    ]

    # Exactly one node may publish odom -> base_footprint. The EKF owns it
    # whenever it runs, so the bridge must stay quiet in that case.
    bridge_publish_tf = PythonExpression(
        ["not ('", ekf, "'.lower() in ('true', '1'))"])

    robot_description = ParameterValue(
        Command(['xacro "', xacro_file, '" is_sim:=false']), value_type=str)

    robot_state_pub = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': False}])

    joint_state_pub = Node(
        package='joint_state_publisher', executable='joint_state_publisher',
        name='joint_state_publisher', output='screen',
        parameters=[{'use_sim_time': False}])

    # -- Jetson ZeroMQ LAN bridge (the default path) ------------------------
    jetson_bridge = Node(
        executable=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jetson_bridge_node_lean.py'),
        name='jetson_bridge_node', output='screen',
        condition=IfCondition(use_jetson),
        parameters=[
            # gyro_params.yaml FIRST so the inline dict can still override it.
            os.path.join(pkg_hw, 'config', 'gyro_params.yaml'),
            {
                'jetson_ip': resolved_jetson_ip,
                'telemetry_port': 5555,
                'cmd_port': 5556,
                'laser_frame_id': 'laser_link',
                'base_frame_id': 'base_footprint',
                'odom_frame_id': 'odom',
                'imu_frame_id': 'imu_link',
                'publish_tf': ParameterValue(bridge_publish_tf, value_type=bool),
                'auto_arm': True,
                'use_sim_time': False,
            },
        ])

    # -- Direct USB serial mode (use_jetson:=false) -------------------------
    lidar = Node(
        package='ldlidar_stl_ros2', executable='ldlidar_stl_ros2_node',
        name='ldlidar_node', output='screen',
        condition=UnlessCondition(use_jetson),
        parameters=[{
            'product_name': 'LDLiDAR_LD19', 'topic_name': 'scan',
            'frame_id': 'laser_link', 'port_name': lidar_port,
            'port_baudrate': 230400, 'laser_scan_dir': True,
            'enable_angle_crop_func': False,
            'angle_crop_min': 135.0, 'angle_crop_max': 225.0,
        }])

    stm32_bridge = Node(
        package='perceptron_hardware', executable='stm32_bridge_node',
        name='stm32_bridge_node', output='screen',
        condition=IfCondition(PythonExpression(
            ["'", stm32, "'.lower() in ('true', '1') and not ('",
             use_jetson, "'.lower() in ('true', '1'))"])),
        parameters=[os.path.join(pkg_hw, 'config', 'hardware_params.yaml'),
                    {'serial_port': stm32_port, 'use_sim_time': False}])

    battery = Node(
        package='perceptron_hardware', executable='battery_node',
        name='battery_node', output='screen', condition=IfCondition(stm32),
        parameters=[os.path.join(pkg_hw, 'config', 'battery_real.yaml')])

    ekf_node = Node(
        package='robot_localization', executable='ekf_node',
        name='ekf_filter_node', output='screen', condition=IfCondition(ekf),
        parameters=[os.path.join(pkg_bringup, 'config', 'ekf_real.yaml')])

    # map_server holds the saved occupancy grid. Transient-local durability so
    # a node that starts later still receives the map, which is published once
    # rather than repeatedly.
    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml, 'use_sim_time': False,
                     'topic_name': 'map', 'frame_id': 'map'}])

    # set_initial_pose is true in nav2_params.yaml because the SIMULATION
    # launch knows exactly where it spawned the robot. On the real robot there
    # is no such ground truth, and leaving it true starts AMCL confidently at
    # the map origin -- it then spends its particle budget converging on that
    # lie, and an /initialpose arriving later (from RViz, or from
    # aruco_localizer_node) has to fight it back out. Off here, on there.
    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl', output='screen',
        parameters=[os.path.join(pkg_nav, 'config', 'nav2_params.yaml'),
                    {'use_sim_time': False, 'set_initial_pose': False}])

    # Nav2 nodes are lifecycle nodes: they come up unconfigured and do nothing
    # until something transitions them. Without this manager map_server holds
    # no map and AMCL never localizes, in complete silence.
    lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[{'use_sim_time': False, 'autostart': True,
                     'node_names': ['map_server', 'amcl']}])

    # The answer to "where am I?" at startup, which is the one question AMCL
    # cannot answer for itself -- hence set_initial_pose: False above. This node
    # solves a surveyed wall board and hands the result over on /initialpose,
    # then stops: AMCL owns map -> odom and beats this everywhere except at
    # startup. /aruco/relocalize arms another seed.
    #
    # It needs jetson_robot_bridge.py running WITHOUT --no-aruco, because the
    # corner pixels arrive over ZMQ rather than as an image topic. If
    # marker_map.yaml has not been surveyed yet the node logs INERT every ten
    # seconds and changes nothing else -- the robot still localizes by hand.
    aruco_localizer = Node(
        package='perceptron_navigation', executable='aruco_localizer_node',
        name='aruco_localizer_node', output='screen',
        condition=IfCondition(aruco),
        parameters=[os.path.join(pkg_nav, 'config', 'aruco_localization.yaml'),
                    {'use_sim_time': False,
                     'jetson_ip': resolved_jetson_ip,
                     'marker_map_path': marker_map}])

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(rviz), parameters=[{'use_sim_time': False}],
        arguments=['-d', PathJoinSubstitution(
            [FindPackageShare('perceptron_robot_bringup'), 'rviz', 'robot.rviz'])])

    return LaunchDescription(args + [
        robot_state_pub, joint_state_pub, jetson_bridge, lidar, stm32_bridge,
        battery, ekf_node, map_server, amcl, lifecycle, aruco_localizer,
        rviz_node,
    ])
