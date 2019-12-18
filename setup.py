import os
import re
import setuptools
from pathlib import Path
p = Path(__file__)

def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)

version = get_version('pytorch_chemistry')

setuptools.setup(
    name="pytorch_chemistry",
    version=version,
    python_requires='>3.7',
    author="Koji Ono",
    author_email="kbu94982@gmail.com",
    description="Drug Discovery Library based on Deep Learning",
    long_description=(p.parent / 'README.md').open(encoding='utf-8').read(),
    packages=setuptools.find_packages(),
    install_requires=['h5py', 'tqdm'
    ],
    setup_requires=['numpy', 'pytest-runner'],
    tests_require=['pytest-cov', 'pytest-html', 'pytest'],
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
