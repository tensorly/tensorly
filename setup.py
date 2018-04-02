import re
import ast

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

# Grab the version
_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('tensorly/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

def readme():
    with open('README.rst') as f:
        return f.read()

config = {
    'name': 'tensorly',
    'packages': find_packages(exclude=['doc']),
    'description': 'Tensor learning in Python.',
    'long_description': readme(),
    'author': 'Jean Kossaifi',
    'author_email': 'jean.kossaifi@gmail.com',
    'version': version,
    'url': 'https://github.com/tensorly/tensorly',
    'download_url': 'https://github.com/tensorly/tensorly/tarball/' + version,
    'install_requires': ['numpy', 'scipy'],
    'license': 'Modified BSD',
    'scripts': [],
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],
}

setup(**config)
