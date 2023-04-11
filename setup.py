# setup file for pip installation
from setuptools import setup, find_packages

setup(
    name='gym_quad',
    version='0.1.0',
    description='Quadrotor environment based on Gym API and Pybullet',
    author='Chaoyi Pan',
    packages=find_packages(),
    install_requires=[
        'gym',
        'numpy',
        'matplotlib',
        'seaborn',
        'icecream',
        'pybullet',
    ],
)