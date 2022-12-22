import os
from setuptools import setup, find_packages

import environmental


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='environmental',
    version=environmental.__version__,
    description="Exceptional Model Mining (EMM)",
    author="Oren Zeev-Ben-Mordehai",
    author_email='zbenmo@gmail.com',
    url="https://github.com/zbenmo/environmental",
    maintainer='zbenmo@gmail.com',
    packages=find_packages(),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    install_requires=[
        'gym',
    ],
    extras_require={
        'examples': [
            'jupyterlab',
            'matplotlib'
        ],
        'tests': [
            'pytest'
        ]
    },
    license=read("LICENSE"),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)