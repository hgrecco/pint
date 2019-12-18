#!/usr/bin/env python

from setuptools import setup


def read(filename):
    with open(filename) as fh:
        return fh.read()


long_description = "\n\n".join([read("README"), read("AUTHORS"), read("CHANGES")])
__doc__ = long_description

setup(
    name="Pint",
    use_scm_version=True,
    description="Physical quantities module",
    long_description=long_description,
    keywords="physical quantities unit conversion science",
    author="Hernan E. Grecco",
    author_email="hernan.grecco@gmail.com",
    url="https://github.com/hgrecco/pint",
    test_suite="pint.testsuite.testsuite",
    zip_safe=True,
    packages=["pint"],
    package_data={"pint": ["default_en.txt", "constants_en.txt"]},
    include_package_data=True,
    license="BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6",
    install_requires=["setuptools"],
    setup_requires=["setuptools", "setuptools_scm"],
    extras_require={
        "numpy": ["numpy >= 1.14"],
        "uncertainties": ["uncertainties >= 3.0"],
    },
)
