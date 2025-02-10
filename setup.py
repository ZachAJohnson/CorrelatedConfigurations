# setup.py
from setuptools import setup, find_packages

setup(
    name='CorrelatedConfigurations',
    version='0.1.0',
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=['numpy', 'scipy'],
    author='Zachary A. Johnson',
    description='Generates configurations of particles for use in initializing molecular dynamics simulations.',
    url='https://github.com/ZachAJohnson/CorrelatedConfigurations',
)
