try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import tensorly
version = tensorly.__version__

config = {
    'name': 'tensorly',
    'packages': find_packages(),
    'description': 'Tensor learning in Python.',
    'author': 'Jean Kossaifi',
    'author_email': 'jean.kossaifi@gmail.com',
    'version': version,
    'url': 'https://github.com/tensorly/tensorly',
    'download_url': 'https://github.com/tensorly/tensorly.tarball' + version,
    'install_requires': ['numpy', 'scipy'],
    'scripts': []
}

setup(**config)
