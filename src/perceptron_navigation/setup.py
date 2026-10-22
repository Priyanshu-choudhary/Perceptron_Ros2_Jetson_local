import os
from glob import glob
from setuptools import setup

package_name = 'perceptron_navigation'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'EXPLORATION.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'behavior_trees'),
         glob('behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Perceptron Developer',
    maintainer_email='dev@todo.todo',
    description='SLAM, localisation, Nav2 and waypoint missions for the Perceptron robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cmd_vel_relay = perceptron_navigation.cmd_vel_relay:main',
            'mission_node = perceptron_navigation.mission_node:main',
            'traversability_node = perceptron_navigation.traversability_node:main',
            'mission = perceptron_navigation.mission_client:main',
            'exploration_node = perceptron_navigation.exploration_node:main',
            'aruco_search_detector = perceptron_navigation.aruco_search_detector:main',
            'explore = perceptron_navigation.exploration_client:main',
            'patrol_node = perceptron_navigation.patrol_node:main',
            'patrol = perceptron_navigation.patrol_client:main',
            'wait_for_odom_tf = perceptron_navigation.wait_for_odom_tf:main',
            'aruco_localizer_node = perceptron_navigation.aruco_localizer_node:main',
            'teach_marker_node = perceptron_navigation.teach_marker_node:main',
            'path_overlay_node = perceptron_navigation.path_overlay_node:main',
            'gps_goal_node = perceptron_navigation.gps_goal_node:main',
            'gps_goal = perceptron_navigation.gps_goal_client:main',
            'predictive_speed_limit_node = perceptron_navigation.predictive_speed_limit_node:main',
        ],
    },
)
