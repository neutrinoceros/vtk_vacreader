from setuptools import setup, find_packages
from pathlib import Path
import os

here = Path(__file__).parent.absolute()
project_name = 'vtk_vacreader'

setup(
    name = project_name,
    version        = __import__(project_name).__version__,
    author         = __import__(project_name).__author__,
    author_email   = __import__(project_name).__contact__,
    url = 'https://gitlab.oca.eu/crobert/vtk_vacreader-project',
    description = 'Read vtu files from non-amr simulations with amrvac',
    license = 'GNU',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6'
    ],
    keywords = 'wrapper',
    install_requires = ['vtk>=7.0.0'],
    python_requires = '>=3.6',
    packages = find_packages(),
)
