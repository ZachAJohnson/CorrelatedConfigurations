# setup.py
from setuptools import setup, find_packages

setup(
    name='ConfigurationGeneration',
    version='0.1.0',
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=['numpy', 'scipy'],
    author='Zachary A. Johnson',
    description='A description of my package',
    url='https://github.com/ZachAJohnson/CorrelatedConfigurations',
)
