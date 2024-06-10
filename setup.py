from setuptools import find_packages, setup

setup(
    name='calflow',
    version='0.1',
    python_requires='>=3.8, <3.9',
    packages=find_packages('src'),
    package_dir={'':'src'},
)