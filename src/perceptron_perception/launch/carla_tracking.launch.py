"""Stage 6: multi-object tracking on top of Stages 4-5, against the CARLA bridge.

    # Windows: CarlaUE4.exe.  WSL:
    ros2 run perceptron_carla carla_bridge --crossing 6                        # known pedestrian speeds
    ros2 run perceptron_carla carla_bridge --scenario --crossing 4 --ego-speed 1.5   # robot moving
    ros2 run perceptron_carla carla_bridge --autopilot --traffic 60 --seed 3   # traffic
    ros2 launch perceptron_perception carla_tracking.launch.py                 # eval:=false to skip scoring
    rviz2 -d $(ros2 pkg prefix perceptron_carla)/share/perceptron_carla/rviz/carla.rviz

Tracks are kept in odom: ground truth in the bridge's gt mode; with --tf-mode ekf plus
perceptron_carla sim_localization.launch.py fixes:=true they ride on the Stage 3 EKF.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('perceptron_perception'), 'config', 'carla.yaml')
    node = lambda exe, **kw: Node(package='perceptron_perception', executable=exe, name=exe,  # noqa: E731
                                  output='screen', parameters=[params], **kw)
    return LaunchDescription([
        DeclareLaunchArgument('eval', default_value='true', description='run tracking_eval_node'),
        node('lidar_preprocessor_node'),
        node('object_fusion_node'),
        node('object_tracker_node'),
        node('tracking_eval_node', condition=IfCondition(LaunchConfiguration('eval'))),
    ])
