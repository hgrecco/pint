.. image:: https://img.shields.io/pypi/v/pint.svg
    :target: https://pypi.python.org/pypi/pint
    :alt: Latest Version

.. image:: https://readthedocs.org/projects/pip/badge/
    :target: http://pint.readthedocs.org/
    :alt: Documentation

.. image:: https://img.shields.io/pypi/l/pint.svg
    :target: https://pypi.python.org/pypi/pint
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/pint.svg
    :target: https://pypi.python.org/pypi/pint
    :alt: Python Versions

.. image:: https://travis-ci.org/hgrecco/pint.svg?branch=master
    :target: https://travis-ci.org/hgrecco/pint
    :alt: CI

.. image:: https://coveralls.io/repos/github/hgrecco/pint/badge.svg?branch=master 
    :target: https://coveralls.io/github/hgrecco/pint?branch=master
    :alt: Coverage

.. image:: https://readthedocs.org/projects/pint/badge/
    :target: http://pint.readthedocs.org/
    :alt: Docs


Pint: makes units easy
======================

Pint is a Python package to define, operate and manipulate physical
quantities: the product of a numerical value and a unit of measurement.
It allows arithmetic operations between them and conversions from and
to different units.

It is distributed with a comprehensive list of physical units, prefixes
and constants. Due to its modular design, you can extend (or even rewrite!)
the complete list without changing the source code. It supports a lot of
numpy mathematical operations **without monkey patching or wrapping numpy**.

It has a complete test coverage. It runs in Python 2.6 and 3.X
with no other dependency. It is licensed under BSD.

It is extremely easy and natural to use:

.. code-block:: python

    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> 3 * ureg.meter + 4 * ureg.cm
    <Quantity(3.04, 'meter')>

and you can make good use of numpy if you want:

.. code-block:: python

    >>> import numpy as np
    >>> [3, 4] * ureg.meter + [4, 3] * ureg.cm
    <Quantity([ 3.04  4.03], 'meter')>
    >>> np.sum(_)
    <Quantity(7.07, 'meter')>


Quick Installation
------------------

To install Pint, simply:

.. code-block:: bash

    $ pip install pint

and then simply enjoy it!


Documentation
-------------

Full documentation is available at http://pint.readthedocs.org/


Design principles
-----------------

Although there are already a few very good Python packages to handle physical
quantities, no one was really fitting my needs. Like most developers, I programed
Pint to scratch my own itches.

- Unit parsing: prefixed and pluralized forms of units are recognized without
  explicitly defining them. In other words: as the prefix *kilo* and the unit *meter*
  are defined, Pint understands *kilometers*. This results in a much shorter and
  maintainable unit definition list as compared to other packages.

- Standalone unit definitions: units definitions are loaded from simple and
  easy to edit text file. Adding and changing units and their definitions does
  not involve changing the code.

- Advanced string formatting: a quantity can be formatted into string using
  PEP 3101 syntax. Extended conversion flags are given to provide latex and pretty
  formatting.

- Small codebase: small and easy to maintain with a flat hierarchy.

- Dependency free: it depends only on Python and its standard library.

- Python 2 and 3: A single codebase that runs unchanged in Python 2.6+ and Python 3.0+.

- Advanced NumPy support: While NumPy is not a requirement for Pint,
  when available ndarray methods and ufuncs can be used in Quantity objects.
