#!/usr/bin/env python

# from setuptools import setup
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# d = generate_distutils_setup(
#     packages=[''],
#     package_dir={'': '.'} 
# )

# setup(**d)


setup(
    name='skills_manager',
    version='0.0.0',
    packages=['skills_manager'],
    package_dir={'skills_manager': 'src/' + 'skills_manager'},
    install_requires=['setuptools'],
    author='',
    author_email='your.email@example.com',
    description='Example Python ROS package',
    license='MIT',
    tests_require=['pytest'],
)