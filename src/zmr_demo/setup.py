from setuptools import find_packages, setup # type: ignore

import os
from glob import glob

package_name = 'zmr_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Include launch files
        (os.path.join('share', package_name,'launch'), glob('launch/*.launch.py')),
        # Include rviz files
        (os.path.join('share', package_name,'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ze',
    maintainer_email='zhze.zhang@gmail.com',
    description='Demos for ZMR',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
