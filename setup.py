#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import subprocess

from setuptools import setup


def read(filename):
    return codecs.open(filename, encoding='utf-8').read()


long_description = '\n\n'.join([read('README'),
                                read('AUTHORS'),
                                read('CHANGES')])

__doc__ = long_description


version = "0.10"
RELEASE_VERSION = False

if not RELEASE_VERSION:
    # Append incremental version number from git
    try:
        version += (
            ".dev"
            + subprocess.check_output(["git", "rev-list", "--count", "HEAD"])
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # The .git directory has been removed (likely by setup.py sdist),
        # and/or git is not installed
        version += ".dev0"


setup(
    name='Pint',
    version=version,
    description='Physical quantities module',
    long_description=long_description,
    keywords='physical quantities unit conversion science',
    author='Hernan E. Grecco',
    author_email='hernan.grecco@gmail.com',
    url='https://github.com/hgrecco/pint',
    test_suite='pint.testsuite.testsuite',
    zip_safe=True,
    packages=['pint'],
    package_data={
        'pint': ['default_en.txt',
                 'constants_en.txt']
      },
    include_package_data=True,
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
    install_requires=['setuptools'],
    extras_require={
        'numpy': ['numpy >= 1.14'],
        'uncertainties': ['uncertainties >= 3.0'],
    },
)
