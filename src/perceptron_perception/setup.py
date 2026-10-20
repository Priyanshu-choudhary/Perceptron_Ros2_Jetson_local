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
        ],
    },
)
