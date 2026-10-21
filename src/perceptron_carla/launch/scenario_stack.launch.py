"""The stack and evaluators for one Stage 9 scenario (started by scenario_suite).

    ros2 launch perceptron_carla scenario_stack.launch.py stack:=perception out_dir:=/tmp/run
    ros2 launch perceptron_carla scenario_stack.launch.py stack:=localization out_dir:=/tmp/run

perception:   Stages 4-8 (preprocessor, fusion, tracker, predictor, costmap) and the five
              perception evaluators
localization: the Stage 3 EKFs with the fixes on, and localization_eval_node
Every evaluator writes its since-start summary to <out_dir>/<node name>.json.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

PERCEPTION = ['lidar_preprocessor_node', 'object_fusion_node', 'object_tracker_node',
              'trajectory_predictor_node', 'costmap_node']
PERCEPTION_EVAL = ['cloud_eval_node', 'fusion_eval_node', 'tracking_eval_node',
                   'prediction_eval_node', 'costmap_eval_node']


def _nodes(context):
    stack = context.launch_configurations['stack']
    out = os.path.expanduser(context.launch_configurations['out_dir'])
    if stack == 'perception':
        params = os.path.join(get_package_share_directory('perceptron_perception'), 'config', 'carla.yaml')
        nodes = [Node(package='perceptron_perception', executable=n, name=n, output='screen',
                      parameters=[params]) for n in PERCEPTION]
        nodes += [Node(package='perceptron_perception', executable=n, name=n, output='screen',
                       parameters=[params, {'use_sim_time': True, 'report_period': 10.0,
                                            'summary_file': os.path.join(out, n + '.json')}])
                  for n in PERCEPTION_EVAL]
        return nodes
    if stack == 'localization':
        launch = os.path.join(get_package_share_directory('perceptron_carla'), 'launch',
                              'sim_localization.launch.py')
        return [IncludeLaunchDescription(PythonLaunchDescriptionSource(launch), launch_arguments={
            'fixes': 'true', 'eval': 'true', 'report_period': '10.0',
            'summary_file': os.path.join(out, 'localization_eval_node.json')}.items())]
    raise RuntimeError('stack must be perception or localization, not %r' % stack)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('stack', default_value='perception', description='perception | localization'),
        DeclareLaunchArgument('out_dir', default_value='/tmp/carla_scenario',
                              description='where the evaluators write their JSON summaries'),
        OpaqueFunction(function=_nodes),
    ])
