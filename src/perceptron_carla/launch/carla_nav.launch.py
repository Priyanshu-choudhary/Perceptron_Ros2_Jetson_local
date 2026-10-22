"""Stage 11: drive the CARLA car to a GPS goal with Nav2 on the Stage 4-8 perception stack.

    # Windows: CarlaUE4.exe.  WSL terminal 1 -- an empty world first, Nav2 drives the car:
    ros2 run perceptron_carla carla_bridge --control ros
    # WSL terminal 2:
    ros2 launch perceptron_carla carla_nav.launch.py                 # perception + Nav2 + gps_goal_node
    # WSL terminal 3:
    rviz2 -d $(ros2 pkg prefix perceptron_carla)/share/perceptron_carla/rviz/carla.rviz
    # WSL terminal 4 -- a goal (or click "Publish Point" / "2D Goal Pose" in RViz):
    ros2 run perceptron_navigation gps_goal --ahead 40
    ros2 run perceptron_navigation gps_goal 19.1338 72.9139

controller_server -> /cmd_vel_nav -> collision_monitor -> /cmd_vel -> carla_bridge --control ros -> the car;
predictive_speed_limit_node -> /speed_limit -> controller_server (yield to predicted crossings).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('perceptron_carla')
    params = os.path.join(share, 'config', 'nav2_carla.yaml')
    bt = os.path.join(share, 'config', 'carla_nav_bt.xml')
    bt_poses = os.path.join(share, 'config', 'carla_nav_through_poses_bt.xml')
    perception = os.path.join(get_package_share_directory('perceptron_perception'), 'launch',
                              'carla_costmap.launch.py')

    def nav2(pkg, exe, name=None, extra=None, remaps=None):
        return Node(package=pkg, executable=exe, name=name or exe, output='screen',
                    parameters=[params] + ([extra] if extra else []), remappings=remaps or [])

    return LaunchDescription([
        DeclareLaunchArgument('perception', default_value='true',
                              description='start Stages 4-8 (preprocessor ... costmap_node)'),
        DeclareLaunchArgument('road', default_value='false',
                              description='also the road map (needs the semantic camera)'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(perception),
                                 launch_arguments={'eval': 'false', 'road': LaunchConfiguration('road')}.items(),
                                 condition=IfCondition(LaunchConfiguration('perception'))),
        nav2('nav2_controller', 'controller_server', remaps=[('cmd_vel', 'cmd_vel_nav')]),
        nav2('nav2_collision_monitor', 'collision_monitor'),
        nav2('perceptron_navigation', 'predictive_speed_limit_node'),
        nav2('nav2_planner', 'planner_server'),
        nav2('nav2_behaviors', 'behavior_server'),
        nav2('nav2_bt_navigator', 'bt_navigator', extra={'default_nav_to_pose_bt_xml': bt,
                                                             'default_nav_through_poses_bt_xml': bt_poses}),
        # 5 s later: brought up together under load, a lifecycle service response was lost (a
        # Humble DDS race: "failed to send response to /controller_server/change_state") and
        # the manager waited forever with Nav2 never active
        TimerAction(period=5.0, actions=[
            nav2('nav2_lifecycle_manager', 'lifecycle_manager', name='lifecycle_manager_navigation')]),
        nav2('perceptron_navigation', 'gps_goal_node'),
    ])
