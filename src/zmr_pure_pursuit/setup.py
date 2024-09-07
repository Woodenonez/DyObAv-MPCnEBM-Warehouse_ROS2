from setuptools import find_packages, setup # type: ignore

import os
from glob import glob

package_name = 'zmr_pure_pursuit'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ze',
    maintainer_email='zhze.zhang@gmail.com',
    description='A pure pursuit controller for zmr robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_controller_node = zmr_pure_pursuit.pure_pursuit_controller_node:main',
        ],
    },
)
