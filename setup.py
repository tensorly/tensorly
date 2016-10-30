try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import tensorly
version = tensorly.__version__


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
