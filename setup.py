#!/usr/bin/env python
"""
created 1/13/20 

@author DevXl

Setup file for the daedalus project
"""
from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='daedalus',
      version='1.0',
      description='Set of routine attributes/functions for vision science experiments',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Cognitive Neuroscience :: Psychophysics',
      ],
      keywords='funniest joke comedy flying circus',
      url='http://github.com/devxl/daedalus',
      author='DevXI',
      author_email='sharif.saleki@gmail.com',
      packages=['daedalus'],
      install_requires=[
          'psychopy',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
