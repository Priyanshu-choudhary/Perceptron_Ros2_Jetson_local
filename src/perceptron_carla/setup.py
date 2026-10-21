import os
from glob import glob
from setuptools import setup

package_name = 'perceptron_carla'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Perceptron Developer',
    maintainer_email='dev@todo.todo',
    description='CARLA simulator bridge for the Perceptron rover',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'carla_bridge = perceptron_carla.carla_bridge:main',
            'localization_eval_node = perceptron_carla.localization_eval_node:main',
            'sensor_check = perceptron_carla.sensor_check:main',
            'scenario_suite = perceptron_carla.scenario_suite:main',
        ],
    },
)
