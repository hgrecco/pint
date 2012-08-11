#!/usr/bin/env python

from distutils.core import setup

with open('README') as file:
    long_description = file.read()

setup(
    name='Pint',
    version='0.1.2',
    description='Physical quantities module',
    long_description=long_description,
    author='Hernan E. Grecco',
    author_email='hernan.grecco@gmail.com',
    url='https://github.com/hgrecco/pint',
    
    packages=['pint'],
    package_data={
        'pint': [
        'default_en.txt']},
    
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
      'Topic :: Software Development :: Libraries'
    ])
