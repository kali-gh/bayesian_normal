from setuptools import setup, find_packages


from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

setup(
    name="bayesian_normal",
    version="0.1",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules = [splitext(basename(path))[0] for path in glob('src/*.py')],
)