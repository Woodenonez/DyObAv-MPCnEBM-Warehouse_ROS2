from setuptools import find_packages, setup # type: ignore

import os
from glob import glob

package_name = 'zmr_description'

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

        (os.path.join('share', package_name,'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name,'rviz'), glob('rviz/*.rviz')),
        *generate_data_files('./urdf', os.path.join('share', package_name, 'urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ze',
    maintainer_email='zhze.zhang@gmail.com',
    description='Description of an autonomous mobile robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rotate_wheel = zmr_description.rotate_wheel:main',
            'spawn_robot = zmr_description.spawn_robot:main',
        ],
    },
)
