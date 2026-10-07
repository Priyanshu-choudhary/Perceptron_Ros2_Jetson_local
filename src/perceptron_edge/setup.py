import os
from glob import glob

from setuptools import setup

package_name = 'perceptron_edge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.xml')),
        (os.path.join('share', package_name, 'behavior_trees'), glob('behavior_trees/*.xml')),
        (os.path.join('share', package_name, 'host'), [f for f in glob('host/*') if os.path.isfile(f)]),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'dock = perceptron_edge.dock:main',
            'dock_guard = perceptron_edge.dock_guard:main',
            'gps_waypoint = perceptron_edge.gps_waypoint_client:main',
        ],
    },
    zip_safe=True,
    maintainer='Priyanshu-choudhary',
    maintainer_email='hcjha05@gmail.com',
    description='Headless, optimised SLAM + Nav2 bringup and charger docking for the Jetson Nano',
    license='Apache-2.0',
)
