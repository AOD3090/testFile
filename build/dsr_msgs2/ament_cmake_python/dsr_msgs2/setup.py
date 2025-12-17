from setuptools import find_packages
from setuptools import setup

setup(
    name='dsr_msgs2',
    version='1.1.0',
    packages=find_packages(
        include=('dsr_msgs2', 'dsr_msgs2.*')),
)
