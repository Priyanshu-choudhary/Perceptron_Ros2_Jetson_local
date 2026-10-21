import os
from glob import glob
from setuptools import setup

package_name = 'perceptron_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Perceptron Developer',
    maintainer_email='dev@todo.todo',
    description='3D lidar + camera perception for the Perceptron robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_camera_overlay_node = perceptron_perception.lidar_camera_overlay_node:main',
            'projection_check_node = perceptron_perception.projection_check_node:main',
            'lidar_preprocessor_node = perceptron_perception.lidar_preprocessor_node:main',
            'cloud_eval_node = perceptron_perception.cloud_eval_node:main',
            'object_fusion_node = perceptron_perception.object_fusion_node:main',
            'fusion_eval_node = perceptron_perception.fusion_eval_node:main',
            'object_tracker_node = perceptron_perception.object_tracker_node:main',
            'tracking_eval_node = perceptron_perception.tracking_eval_node:main',
            'trajectory_predictor_node = perceptron_perception.trajectory_predictor_node:main',
            'prediction_eval_node = perceptron_perception.prediction_eval_node:main',
            'costmap_node = perceptron_perception.costmap_node:main',
            'costmap_eval_node = perceptron_perception.costmap_eval_node:main',
        ],
    },
)
