from setuptools import find_packages, setup # type: ignore

import os
from glob import glob

package_name = 'map_description'

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

        # Add map_description/data/map.json to the package
        (os.path.join('share', package_name, 'data'), glob('data/*.json')),
        # Add launch files to the package
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Add rviz files to the package
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        # Add model files to the package
        *generate_data_files('./model', os.path.join('share', package_name, 'model')),
        # Add world files to the package
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools', 'shapely'],
    zip_safe=True,
    maintainer='ze',
    maintainer_email='zhze.zhang@gmail.com',
    description='Read pre-defined map and publish it to a certain type of map topic',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'geometric_map_node = map_description.geometric_map_node:main',
        ],
    },
)
