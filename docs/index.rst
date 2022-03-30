:orphan:

Pint: makes units easy
======================

.. image:: _static/logo-full.jpg
   :alt: Pint: **physical quantities**
   :class: floatingflask

Pint is a Python package to define, operate and manipulate **physical quantities**:
the product of a numerical value and a unit of measurement. It allows
arithmetic operations between them and conversions from and to different units.

It is distributed with a `comprehensive list of physical units, prefixes and constants`_.
Due to its modular design, you can extend (or even rewrite!) the complete list
without changing the source code. It supports a lot of numpy mathematical
operations **without monkey patching or wrapping numpy**.

It has a complete test coverage. It runs in Python 3.8+ with no other
dependencies. It is licensed under a `BSD 3-clause style license`_.

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

See the :ref:`Tutorial` for more help getting started.

Quick Installation
------------------

To install Pint, simply:

.. code-block:: bash

    $ pip install pint

or utilizing conda, with the conda-forge channel:

.. code-block:: bash

    $ conda install -c conda-forge pint

and then simply enjoy it!

(See :ref:`Installation <getting>` for more detail.)


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
(``fraction``, ``float``, ``decimal``, ``numpy.ndarray``, etc). NumPy_ is not
required, but is supported.

**Awesome NumPy integration**: When you choose to use a NumPy_ ndarray, its methods and
ufuncs are supported including automatic conversion of units. For example
``numpy.arccos(q)`` will require a dimensionless ``q`` and the units of the output
quantity will be radian.

**Uncertainties integration**:  transparently handles calculations with
quantities with uncertainties (like 3.14±0.01) meter via the `uncertainties
package`_.

**Handle temperature**: conversion between units with different reference
points, like positions on a map or absolute temperature scales.

**Dependency free**: it depends only on Python and its standard library. It interacts with other packages
like numpy and uncertainties if they are installed

**Pandas integration**: The `pint-pandas`_ package makes it possible to use Pint with Pandas.
Operations on DataFrames and between columns are units aware, providing even more convenience for users
of Pandas DataFrames. For full details, see the `pint-pandas Jupyter notebook`_.


When you choose to use a NumPy_ ndarray, its methods and
ufuncs are supported including automatic conversion of units. For example
``numpy.arccos(q)`` will require a dimensionless ``q`` and the units
of the output quantity will be radian.


User Guide
----------

.. toctree::
    :maxdepth: 1

    getting
    tutorial
    defining-quantities
    formatting
    numpy
    nonmult
    log_units
    wrapping
    plotting
    serialization
    pitheorem
    contexts
    measurement
    defining
    performance
    systems
    currencies
    pint-convert

More information
----------------

.. toctree::
    :maxdepth: 1

    developers_reference
    contributing
    faq


One last thing
--------------

.. epigraph::

 The MCO MIB has determined that the root cause for the loss of the MCO spacecraft was the failure to use metric     units in the coding of a ground software file, “Small Forces,” used in trajectory models. Specifically, thruster performance data in English units instead of metric units was used in the software application code titled SM_FORCES (small forces). The output from the SM_FORCES application code as required by a MSOP Project Software Interface Specification (SIS) was to be in metric units of Newtonseconds (N-s). Instead, the data was reported in English units of pound-seconds (lbf-s). The Angular Momentum Desaturation (AMD) file contained the output data from the SM_FORCES software. The SIS, which was not followed, defines both the format and units of the AMD file generated by ground-based computers. Subsequent processing of the data from AMD file by the navigation software algorithm therefore, underestimated the effect on the spacecraft trajectory by a factor of 4.45, which is the required conversion factor from force in pounds to Newtons. An erroneous trajectory was computed using this incorrect data.

            `Mars Climate Orbiter Mishap Investigation Phase I Report`
            `PDF <https://llis.nasa.gov/llis_lib/pdf/1009464main1_0641-mr.pdf>`_



.. _`comprehensive list of physical units, prefixes and constants`: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
.. _`uncertainties package`: https://pythonhosted.org/uncertainties/
.. _`NumPy`: http://www.numpy.org/
.. _`PEP 3101`: https://www.python.org/dev/peps/pep-3101/
.. _`Babel`: http://babel.pocoo.org/
.. _`pint-pandas`: https://github.com/hgrecco/pint-pandas
.. _`pint-pandas Jupyter notebook`: https://github.com/hgrecco/pint-pandas/blob/master/notebooks/pint-pandas.ipynb
.. _`BSD 3-clause style license`: https://github.com/hgrecco/pint/blob/master/LICENSE
