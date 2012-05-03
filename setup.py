#!/usr/bin/env python

from distutils.core import setup

setup(name='Pint',
      version='0.1',
      description='Physical quantities module',
      author='Hernan E. Grecco',
      author_email='hernan.grecco@gmail.com',
      url='https://github.com/hgrecco/pint',
      packages=['pint', ],
      license='BSD',
      data_files=[('lib/pint/', ('pint/default_en.txt', )),]
     )

