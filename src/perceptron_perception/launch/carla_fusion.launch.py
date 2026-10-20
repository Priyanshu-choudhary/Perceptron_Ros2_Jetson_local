"""Stage 5: camera-lidar object fusion on top of Stage 4, against the CARLA bridge.

    # Windows: CarlaUE4.exe.  WSL:
    ros2 run perceptron_carla carla_bridge --crossing 6                         # moving people, known speeds
    ros2 run perceptron_carla carla_bridge --autopilot --traffic 30 --pedestrians 40
    ros2 launch perceptron_perception carla_fusion.launch.py                    # eval:=false to skip scoring
    ros2 run rqt_image_view rqt_image_view /perception/objects_image

detections come from /carla/ground_truth/boxes_2d (set detections_topic in config/carla.yaml
for your own detector). The bridge needs --boxes (default on) for those, and --lidar-gt for
the scoring.
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
    ev = IfCondition(LaunchConfiguration('eval'))
    return LaunchDescription([
        DeclareLaunchArgument('eval', default_value='true', description='run the ground-truth scoring nodes'),
        Node(package='perceptron_perception', executable='lidar_preprocessor_node',
             name='lidar_preprocessor_node', output='screen', parameters=[params]),
        Node(package='perceptron_perception', executable='object_fusion_node',
             name='object_fusion_node', output='screen', parameters=[params]),
        Node(package='perceptron_perception', executable='fusion_eval_node', name='fusion_eval_node',
             output='screen', parameters=[params], condition=ev),
    ])
