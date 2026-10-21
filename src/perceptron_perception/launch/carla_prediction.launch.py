"""Stage 7: trajectory prediction on top of Stages 4-6, against the CARLA bridge.

    # Windows: CarlaUE4.exe.  WSL:
    ros2 run perceptron_carla carla_bridge --crossing 6                          # walkers, known speeds
    ros2 run perceptron_carla carla_bridge --scenario --walkers 0 --ego-speed 1.0
    ros2 run perceptron_carla carla_bridge --autopilot --traffic 60 --pedestrians 40 --seed 3
    ros2 launch perceptron_perception carla_prediction.launch.py                 # eval:=false to skip scoring
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
    node = lambda exe, **kw: Node(package='perceptron_perception', executable=exe, name=exe,  # noqa: E731
                                  output='screen', parameters=[params], **kw)
    ev = IfCondition(LaunchConfiguration('eval'))
    return LaunchDescription([
        DeclareLaunchArgument('eval', default_value='true', description='run prediction_eval_node'),
        node('lidar_preprocessor_node'),
        node('object_fusion_node'),
        node('object_tracker_node'),
        node('trajectory_predictor_node'),
        node('prediction_eval_node', condition=ev),
    ])
