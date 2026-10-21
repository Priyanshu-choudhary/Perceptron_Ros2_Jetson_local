"""Stage 8: the whole perception stack (Stages 4-8) feeding the rolling 3D costmap.

    # Windows: CarlaUE4.exe.  WSL terminal 1 -- the car drives itself through traffic:
    ros2 run perceptron_carla carla_bridge --autopilot --traffic 40 --pedestrians 80 --no-semantic
    # WSL terminal 2:
    ros2 launch perceptron_perception carla_costmap.launch.py            # eval:=false to skip scoring
    # WSL terminal 3:
    rviz2 -d $(ros2 pkg prefix perceptron_carla)/share/perceptron_carla/rviz/carla.rviz

lidar_preprocessor (4) -> object_fusion (5) -> object_tracker (6) -> trajectory_predictor (7)
-> costmap (8), plus costmap_eval scoring every layer against CARLA ground truth.
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
        DeclareLaunchArgument('eval', default_value='true', description='run costmap_eval_node'),
        node('lidar_preprocessor_node'),
        node('object_fusion_node'),
        node('object_tracker_node'),
        node('trajectory_predictor_node'),
        node('costmap_node'),
        node('costmap_eval_node', condition=IfCondition(LaunchConfiguration('eval'))),
    ])
