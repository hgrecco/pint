---
title: 'Pint: Operate and manipulate physical quantities in Python'
tags:
  - units
  - python
authors:
  - name: Hector Grecco
    orcid: ---
    affiliation:
  - name: Author Banana
    orcid: ---
    affiliation:
  - name: Author Cherry
    orcid: ---
    affiliation:
affiliations:
  - name: Institute 1
    index: 1
  - name: Institute 2
    index: 2
date: TBD
bibliography: paper.bib
output: pdf_document
---

# Summary

Pint is a Python package to define, operate and manipulate physical quantities: the product of a numerical value and a unit of measurement. It allows arithmetic operations between them and conversions from and to different units.

It is distributed with a comprehensive list of physical units, prefixes and constants that can be extended or rewritten without changing the source code. Pint natively supports Numpy and uncertainties, and additional modules for pandas and xarray integrations.

# Statement of need

Python is commonly used for scientific data analysis but does not natively provide unit support. Pint provides Quantity objects that store numeric data such as ints, floats or arrays and their units, and propogates or converts units when performing arithmetic operations. This removes the need for researchers to keep track of units and conversion factors, significantly simplifying numerical analysis and reducing the likelihood of errors. 

Pint and astropy.units are the two widely used python units libraries. [https://sci.bban.top/pdf/10.1145/3276604.3276613.pdf#page=12&zoom=100,72,216] Both  are mature libraries with over 10 years of development and have similar functionality. Pint's main advantages are its simple unit definition file, and integrations with Pandas, Xarray and uncertainties.

## Key features

### Unit parsing

Prefixed and pluralized forms of units are recognized without explicitly defining them. In other words: as the prefix kilo and the unit meter are defined, Pint understands kilometers. This results in a much shorter and maintainable unit definition list as compared to other packages.

### Standalone unit definitions

Units definitions are loaded from a text file which is simple and easy to edit. Adding and changing units and their definitions does not involve changing the code. The default file contains over 600 commonly used constants, units and abbreviations. Units can also be defined programatically.

### Advanced string formatting

A quantity can be formatted into string using PEP 3101 https://www.python.org/dev/peps/pep-3101/ syntax. Extended conversion flags are given to provide symbolic, LaTeX and pretty formatting. Unit name translation is available if Babel  http://babel.pocoo.org/ is installed.

### Dimensionality checking

Pint detects operations on quantities that do not make physcial sense and raises a `DimensionalityError`. Examples include adding a length to a mass, or taking the exponential of a quantity which is not dimensionless.

### Temperature & sound units.

Pint handles conversion between units with different reference points, like positions on a map or absolute temperature scales. Logarathimic units, such as for sound pressure level (SPL), are also supported. Pint handles arithmetic operations for these units and prevents potential errors when operations are ambiguous by raising an OffsetUnitCalculusError. For example, when adding 10 °C + 100 °C two different result are reasonable depending on the context, 110 °C  or 383.15 °C.

### Native integration with NumPy, matplotlib and uncertainties

NumPy ndarrays can be used as the numerical value of a quantity, and its methods and ufuncs are supported including automatic conversion of units. For example numpy.arccos(q) will require a dimensionless q and the units of the output quantity will be radian.

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations. [https://matplotlib.org/stable/] Pint supports matplotlib's unit API, which automatically labels plot axes with units and converts plotted data to consitent units. 

The uncertainties package takes the pain and complexity out of uncertainty calculations. [https://github.com/lmfit/uncertainties] Pint provides `Measurement` objects to propogate units and uncertainty in calculations.

### Integrations for pandas and Xarray

pint-pandas [https://github.com/hgrecco/pint-pandas] provides integration with pandas, 'a fast, powerful, flexible and easy to use open source data analysis and manipulation tool' [pandas.pydata.org] and pint-xarray[https://github.com/xarray-contrib/pint-xarray/] provides integration with xarray, which 'makes working with labelled multi-dimensional arrays in Python simple, efficient, and fun!' [https://docs.xarray.dev/en/stable/]


### Contexts

Contexts allow conversion between quantities of incompatable dimensions based on some pre-established (physical) relationships. For example, in spectroscopy you need to convert from wavelength ([length]) to frequency ([1/time]), but this will fail due to the different dimensions. A context containing the relation 'frequency = speed_of_light / wavelength' can be used for this conversion.

### Systems

The units used to define dimensions are the 'base units', and pint provides a function to convert a quantity to its base units. Pint allows the base units to be changed through Systems, which are a group of units. A researcher can change the System to another System, such as cgs, imperial, atomic or Planck, which may be more desirable than using mks, or they may define their a custom system of units.

## Publications using Pint

To be written in full (there are likely heaps):

- pymagicc
- openscm
- fair (?)

# References

- software archive
