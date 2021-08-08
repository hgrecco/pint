.. _measurement:


Using Measurements
==================

If you have the `Uncertainties package`_ installed, you can use Pint to keep
track of measurements with specified uncertainty, and not just exact physical
quantities.

Measurements are the combination of two quantities: the mean value and the error
(or uncertainty). The easiest ways to generate a measurement object is from a
quantity using the ``plus_minus()`` method.

.. doctest::
   :skipif: not_installed['uncertainties']

   >>> import numpy as np
   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry()
   >>> book_length = (20. * ureg.centimeter).plus_minus(2.)
   >>> print(book_length)
   (20.0 +/- 2.0) centimeter

You can inspect the mean value, the absolute error and the relative error:

.. doctest::
   :skipif: not_installed['uncertainties']

   >>> print(book_length.value)
   20.0 centimeter
   >>> print(book_length.error)
   2.0 centimeter
   >>> print(book_length.rel)
   0.1

You can also create a Measurement object giving the relative error:

.. doctest::
   :skipif: not_installed['uncertainties']

   >>> book_length = (20. * ureg.centimeter).plus_minus(.1, relative=True)
   >>> print(book_length)
   (20.0 +/- 2.0) centimeter

Measurements support the same formatting codes as Quantity. For example, to pretty
print a measurement with 2 decimal positions:

.. doctest::
   :skipif: not_installed['uncertainties']

   >>> print('{:.02fP}'.format(book_length))
   (20.00 ± 2.00) centimeter


Mathematical operations with Measurements, return new measurements following
the `Propagation of uncertainty`_ rules.

.. doctest::
   :skipif: not_installed['uncertainties']

   >>> print(2 * book_length)
   (40 +/- 4) centimeter
   >>> width = (10 * ureg.centimeter).plus_minus(1)
   >>> print('{:.02f}'.format(book_length + width))
   (30.00 +/- 2.24) centimeter

.. note:: Only linear combinations are currently supported.


.. _`Propagation of uncertainty`: http://en.wikipedia.org/wiki/Propagation_of_uncertainty
.. _`Uncertainties package`: https://uncertainties-python-package.readthedocs.io/en/latest/
