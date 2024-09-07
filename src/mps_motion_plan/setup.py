from setuptools import find_packages, setup # type: ignore

import os
from glob import glob

package_name = 'mps_motion_plan'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Add data files to the package
        (os.path.join('share', package_name, 'data'), glob('data/*.json')), # graph
        (os.path.join('share', package_name, 'data'), glob('data/*.csv')), # schedule
        # Add launch files to the package
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Add rviz files to the package
        # (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ze',
    maintainer_email='zhze.zhang@gmail.com',
    description='Generate reference path and trajectory given a schedule.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_path_coordinate_node = mps_motion_plan.global_path_coordinate_node:main',
            'schedule_visualize_node = mps_motion_plan.schedule_visualize_node:main',
        ],
    },
)
