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

Pint makes it easy to work with units in Python.
It allows users to operate on quantities in a units aware way.
It also allows users to easily and/or automatically convert units.
This significantly simplifies numerical analysis and greatly reduces the risk of inadvertent units errors appearing in research outputs.

When one considers how data is used by scientific researchers, the need is clear.
For example, when making calculations, it is vital that the units are self-consistent.
Typically programming languages have no units support and hence the researcher must keep track of units themselves, separate from the data.
This is generally dangerous and can easily lead to researchers forgetting to perform unit conversions or losing track of what the units of various quantities are.
This can lead to results which are incorrect by any number of conversion factors.
Whilst these errors are normally picked up in the end, they can require an unreasonable amount of time to find, understand and eradicate.

Pint solves this issue by ensuring that the units are carried with the quantities.
This means that it can check and convert units when making calculations in an automated, consistent way.
Pint's `DimensionalityError` is particularly useful as it alerts the user to any units errors which cannot be resolved (for example, taking the exponential of a quantity which is not dimensionless).
This ensures that users can spend more time on their research and far less time trying to track unit conversions.

It has a number of useful features.
The major two are the ability for users to define their own, custom units and the ability for users to define their own 'contexts', within which they can define their own custom unit conversions.
The utitilty of the first feature is fairly obvious, especially for researchers who operate in fields which have conventions which aren't captured by normal SI concentions.
The second feature allows researchers to follow standard conventions in their field.
For example, in special relativity research one could define a context in which mass and energy where equivalent units and could simply be converted into each other with a factor of c^2 (or 1/c^2 depending on which way you are converting).
This is extremely useful as it allows researchers to convert numbers in a way which is natural for them in their field without forcing the entire library to always have this feature.

Finally, Pint is also compatible with Pandas, a 'widely used data analysis and manipulation library for Python'  [quote from Pandas homepage, reference required?].
This capability makes it the first library to allow users to include units aware quantities in tabular like data.

## Publications using Pint

To be written in full (there are likely heaps):

- pymagicc
- openscm
- fair (?)

# References

- software archive
