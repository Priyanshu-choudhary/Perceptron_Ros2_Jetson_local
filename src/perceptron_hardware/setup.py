import os
from glob import glob
from setuptools import setup

package_name = 'perceptron_hardware'

setup(
    name=package_name,
    version='1.0.0',
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
    description='STM32 Motor Driver and IMU Serial Hardware Bridge for Perceptron robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stm32_bridge_node = perceptron_hardware.stm32_bridge_node:main',
            'battery_node = perceptron_hardware.battery_node:main',
            'ekf_watchdog_node = perceptron_hardware.ekf_watchdog_node:main',
            'detect_ports = perceptron_hardware.port_detect:main',
            'jetson_bridge_node = perceptron_hardware.jetson_bridge_node:main',
            'pixhack_bridge_node = perceptron_hardware.pixhack_bridge_node:main',
        ],
    },
)
