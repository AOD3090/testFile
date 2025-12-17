from setuptools import find_packages
from setuptools import setup

setup(
    name='dsr_mujoco',
    version='0.1.0',
    packages=find_packages(
        include=('dsr_mujoco', 'dsr_mujoco.*')),
)
