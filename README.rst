.. image:: https://img.shields.io/pypi/v/pint.svg
    :target: https://pypi.python.org/pypi/pint
    :alt: Latest Version

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/python/black

.. image:: https://readthedocs.org/projects/pint/badge/
    :target: https://pint.readthedocs.org/
    :alt: Documentation

.. image:: https://img.shields.io/pypi/l/pint.svg
    :target: https://pypi.python.org/pypi/pint
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/pint.svg
    :target: https://pypi.python.org/pypi/pint
    :alt: Python Versions

.. image:: https://github.com/hgrecco/pint/workflows/CI/badge.svg
    :target: https://github.com/hgrecco/pint/actions?query=workflow%3ACI
    :alt: CI

.. image:: https://github.com/hgrecco/pint/workflows/Lint/badge.svg
    :target: https://github.com/hgrecco/pint/actions?query=workflow%3ALint
    :alt: LINTER

.. image:: https://coveralls.io/repos/github/hgrecco/pint/badge.svg?branch=master
    :target: https://coveralls.io/github/hgrecco/pint?branch=master
    :alt: Coverage


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

It has a complete test coverage. It runs in Python 3.8+ with no other dependency.
It is licensed under BSD.

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

or utilizing conda, with the conda-forge channel:

.. code-block:: bash

    $ conda install -c conda-forge pint

and then simply enjoy it!


Documentation
-------------

Full documentation is available at http://pint.readthedocs.org/


Command-line converter
----------------------

A command-line script `pint-convert` provides a quick way to convert between
units or get conversion factors.


Design principles
-----------------

Although there are already a few very good Python packages to handle physical
quantities, no one was really fitting my needs. Like most developers, I
programmed Pint to scratch my own itches.

**Unit parsing**: prefixed and pluralized forms of units are recognized without
explicitly defining them. In other words: as the prefix *kilo* and the unit
*meter* are defined, Pint understands *kilometers*. This results in a much
shorter and maintainable unit definition list as compared to other packages.

**Standalone unit definitions**: units definitions are loaded from a text file
which is simple and easy to edit. Adding and changing units and their
definitions does not involve changing the code.

**Advanced string formatting**: a quantity can be formatted into string using
`PEP 3101`_ syntax. Extended conversion flags are given to provide symbolic,
LaTeX and pretty formatting. Unit name translation is available if Babel_ is
installed.

**Free to choose the numerical type**: You can use any numerical type
(`fraction`, `float`, `decimal`, `numpy.ndarray`, etc). NumPy_ is not required
but supported.

**Awesome NumPy integration**: When you choose to use a NumPy_ ndarray, its methods and
ufuncs are supported including automatic conversion of units. For example
`numpy.arccos(q)` will require a dimensionless `q` and the units of the output
quantity will be radian.

**Uncertainties integration**:  transparently handles calculations with
quantities with uncertainties (like 3.14±0.01 meter) via the `uncertainties
package`_.

**Handle temperature**: conversion between units with different reference
points, like positions on a map or absolute temperature scales.

**Dependency free**: it depends only on Python and its standard library. It interacts with other packages
like numpy and uncertainties if they are installed

**Pandas integration**: Thanks to `Pandas Extension Types`_ it is now possible to use Pint with Pandas. Operations on DataFrames and between columns are units aware, providing even more convenience for users of Pandas DataFrames. For full details, see the `pint-pandas Jupyter notebook`_.


Pint is maintained by a community of scientists, programmers and enthusiasts around the world.
See AUTHORS_ for a complete list.

To review an ordered list of notable changes for each version of a project,
see CHANGES_


.. _Website: http://www.dimensionalanalysis.org/
.. _`comprehensive list of physical units, prefixes and constants`: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
.. _`uncertainties package`: https://pythonhosted.org/uncertainties/
.. _`NumPy`: http://www.numpy.org/
.. _`PEP 3101`: https://www.python.org/dev/peps/pep-3101/
.. _`Babel`: http://babel.pocoo.org/
.. _`Pandas Extension Types`: https://pandas.pydata.org/pandas-docs/stable/extending.html#extension-types
.. _`pint-pandas Jupyter notebook`: https://github.com/hgrecco/pint-pandas/blob/master/notebooks/pandas_support.ipynb
.. _`AUTHORS`: https://github.com/hgrecco/pint/blob/master/AUTHORS
.. _`CHANGES`: https://github.com/hgrecco/pint/blob/master/CHANGES
