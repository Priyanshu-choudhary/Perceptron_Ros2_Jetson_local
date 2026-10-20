"""Lidar-on-camera overlay plus its pixel-error check, against the CARLA bridge.

    # Windows: CarlaUE4.exe running.  WSL:
    python3 carla_min_bridge.py --scenario          # in Carl_sim/scripts
    ros2 launch perceptron_perception carla_overlay.launch.py
    ros2 launch perceptron_perception carla_overlay.launch.py perturb_yaw_deg:=1.0   # must read ~11 px

View /perception/lidar_overlay in RViz (Image display, Reliability: Best Effort).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('perceptron_perception'),
                          'config', 'carla.yaml')
    yaw = LaunchConfiguration('perturb_yaw_deg')
    # A camera yaw is a rotation about optical +Y.
    perturb = {'perturb_rpy_deg': PythonExpression(['[0.0, float(', yaw, '), 0.0]'])}

    return LaunchDescription([
        DeclareLaunchArgument('check', default_value='true',
                              description='also run projection_check_node'),
        DeclareLaunchArgument('perturb_yaw_deg', default_value='0.0',
                              description='fake extrinsic error, to prove the check sees it'),
        Node(package='perceptron_perception', executable='lidar_camera_overlay_node',
             name='lidar_camera_overlay_node', output='screen', parameters=[params, perturb]),
        Node(package='perceptron_perception', executable='projection_check_node',
             name='projection_check_node', output='screen', parameters=[params, perturb],
             condition=IfCondition(LaunchConfiguration('check'))),
    ])
