from setuptools import find_packages, setup # type: ignore

import os
from glob import glob

package_name = 'mmp_motion_predict'

def generate_data_files(source_folder, destination_folder):
    data_files = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_file = os.path.join(root, file)
            destination = os.path.join(destination_folder, os.path.relpath(root, source_folder))
            data_files.append((destination, [source_file]))
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Include all the config files
        (os.path.join('share', package_name,'config'), glob('config/*.yaml')),
        # Include all the launch files
        (os.path.join('share', package_name,'launch'), glob('launch/*.launch.py')),
        # Include all the rviz files
        (os.path.join('share', package_name,'rviz'), glob('rviz/*.rviz')),
        # Include all the data files
        *generate_data_files('./data', os.path.join('share', package_name, 'data')),
        # Include all the models
        *generate_data_files('./model', os.path.join('share', package_name, 'model')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ze',
    maintainer_email='zhze.zhang@gmail.com',
    description='Use neural networks for multimodal motion prediction',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_prediction_node = mmp_motion_predict.motion_prediction_node:main',
            'demo_traj_publisher = mmp_motion_predict.demo_traj_publisher:main',
            'actor_traj_publisher = mmp_motion_predict.actor_traj_publisher:main'
        ],
    },
)
