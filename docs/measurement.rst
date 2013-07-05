.. _measurement:


Using Measurements
==================

Measurements are the combination of two quantities: the mean value and the error (or uncertainty). The easiest ways to generate a measurement object is from a quantity using the `plus_minus` operator.

.. doctest::

   >>> import numpy as np
   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry()
   >>> book_length = (20. * ureg.centimeter).plus_minus(2.)
   >>> print(book_length)
   (20.0 +/- 2.0) centimeter

.. testsetup:: *

   import numpy as np
   from pint import UnitRegistry
   ureg = UnitRegistry()
   Q_ = ureg.Quantity

You can inspect the mean value, the absolute error and the relative error:

.. doctest::

   >>> print(book_length.value)
   20.0 centimeter
   >>> print(book_length.error)
   2.0 centimeter
   >>> print(book_length.rel)
   0.1

You can also create a Measurement object giving the relative error:

.. doctest::

   >>> book_length = (20. * ureg.centimeter).plus_minus(.1, relative=True)
   >>> print(book_length)
   (20.0 +/- 2.0) centimeter

Measurements support the same formatting codes as Quantity. For example, to pretty print a measurement with 2 decimal positions:

.. doctest::

   >>> print('{:.02f!p}'.format(book_length))
   (20.00 ± 2.00) centimeter


Mathematical operations with Measurements, return new measurements following the `Propagation of uncertainty`_ rules.

.. doctest::

   >>> print(2 * book_length)
   (40.0 +/- 4.0) centimeter
   >>> width = (10 * ureg.centimeter).plus_minus(1)
   >>> print('{:.02f}'.format(book_length + width))
   (30.00 +/- 2.24) centimeter

.. note:: only linear combinations are currently supported.


.. _`Propagation of uncertainty`: http://en.wikipedia.org/wiki/Propagation_of_uncertainty
