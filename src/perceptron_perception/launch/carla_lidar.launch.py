"""Stage 4 lidar preprocessing against the CARLA bridge, plus its ground-truth scoring.

    # Windows: CarlaUE4.exe.  WSL:
    ros2 run perceptron_carla carla_bridge --autopilot --traffic 30 --pedestrians 40
    ros2 launch perceptron_perception carla_lidar.launch.py           # eval:=false to skip scoring
    rviz2 -d $(ros2 pkg prefix perceptron_carla)/share/perceptron_carla/rviz/carla.rviz
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
    return LaunchDescription([
        DeclareLaunchArgument('eval', default_value='true', description='run cloud_eval_node'),
        Node(package='perceptron_perception', executable='lidar_preprocessor_node',
             name='lidar_preprocessor_node', output='screen', parameters=[params]),
        Node(package='perceptron_perception', executable='cloud_eval_node', name='cloud_eval_node',
             output='screen', parameters=[params], condition=IfCondition(LaunchConfiguration('eval'))),
    ])
