Defining Quantities
===================

A quantity in Pint is the product of a unit and a magnitude.

Pint supports several different ways of defining physical quantities, including
a powerful string parsing system. These methods are largely interchangeable,
though you may **need** to use the constructor form under certain circumstances
(see :doc:`nonmult` for an example of where the constructor form is required).

By multiplication
-----------------

If you've read the :ref:`Tutorial`, you're already familiar with defining a
quantity by multiplying a ``Unit()`` and a scalar:

.. doctest::

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
    >>> ureg.meter
    <Unit('meter')>
    >>> 30.0 * ureg.meter
    <Quantity(30.0, 'meter')>

This works to build up complex units as well:

.. doctest::

    >>> 9.8 * ureg.meter / ureg.second**2
    <Quantity(9.8, 'meter / second ** 2')>


Using the constructor
---------------------

In some cases it is useful to define :class:`Quantity() <pint.quantity.Quantity>`
objects using it's class constructor. Using the constructor allows you to
specify the units and magnitude separately.

We typically abbreviate that constructor as `Q_` to make it's usage less verbose:

.. doctest::

    >>> Q_ = ureg.Quantity
    >>> Q_(1.78, ureg.meter)
    <Quantity(1.78, 'meter')>

As you can see below, the multiplication and constructor methods should produce
the same results:

.. doctest::

    >>> Q_(30.0, ureg.meter) == 30.0 * ureg.meter
    True
    >>> Q_(9.8, ureg.meter / ureg.second**2)
    <Quantity(9.8, 'meter / second ** 2')>

Quantity can be created with itself, if units is specified ``pint`` will try to convert it to the desired units.
If not, pint will just copy the quantity.

.. doctest::

    >>> length = Q_(30.0, ureg.meter)
    >>> Q_(length, 'cm')
    <Quantity(3000.0, 'centimeter')>
    >>> Q_(length)
    <Quantity(30.0, 'meter')>

Using string parsing
--------------------

Pint includes a powerful parser for detecting magnitudes and units (with or
without prefixes) in strings. Calling the ``UnitRegistry()`` directly
invokes the parsing function:

.. doctest::

    >>> 30.0 * ureg('meter')
    <Quantity(30.0, 'meter')>
    >>> ureg('30.0 meters')
    <Quantity(30.0, 'meter')>
    >>> ureg('3000cm').to('meters')
    <Quantity(30.0, 'meter')>

The parsing function is also available to the ``Quantity()`` constructor and
the various ``.to()`` methods:

.. doctest::

    >>> Q_('30.0 meters')
    <Quantity(30.0, 'meter')>
    >>> Q_(30.0, 'meter')
    <Quantity(30.0, 'meter')>
    >>> Q_('3000.0cm').to('meter')
    <Quantity(30.0, 'meter')>

Or as a standalone method on the ``UnitRegistry``:

.. doctest::

   >>> 2.54 * ureg.parse_expression('centimeter')
   <Quantity(2.54, 'centimeter')>

It is fairly good at detecting compound units:

.. doctest::

    >>> g = ureg('9.8 meters/second**2')
    >>> g
    <Quantity(9.8, 'meter / second ** 2')>
    >>> g.to('furlongs/fortnight**2')
    <Quantity(7.12770743e+10, 'furlong / fortnight ** 2')>

And behaves well when given dimensionless quantities, which are parsed into
their appropriate objects:

.. doctest::

   >>> ureg('2.54')
   2.54
   >>> type(ureg('2.54'))
   <class 'float'>
   >>> Q_('2.54')
   <Quantity(2.54, 'dimensionless')>
   >>> type(Q_('2.54'))
   <class 'pint.quantity.build_quantity_class.<locals>.Quantity'>

.. note:: Pint's rule for parsing strings with a mixture of numbers and
   units is that **units are treated with the same precedence as numbers**.

For example, the units of

.. doctest::

   >>> Q_('3 l / 100 km')
   <Quantity(0.03, 'kilometer * liter')>

may be unexpected at first but, are a consequence of applying this rule. Use
brackets to get the expected result:

.. doctest::

   >>> Q_('3 l / (100 km)')
   <Quantity(0.03, 'liter / kilometer')>

Special strings for NaN (Not a Number) and inf(inity) are also handled in a case-insensitive fashion.
Note that, as usual, NaN != NaN.

.. doctest::

   >>> Q_('inf m')
   <Quantity(inf, 'meter')>
   >>> Q_('-INFINITY m')
   <Quantity(-inf, 'meter')>
   >>> Q_('nan m')
   <Quantity(nan, 'meter')>
   >>> Q_('NaN m')
   <Quantity(nan, 'meter')>

.. note:: Since version 0.7, Pint **does not** use eval_ under the hood.
   This change removes the `serious security problems`_ that the system is
   exposed to when parsing information from untrusted sources.

.. _eval: http://docs.python.org/3/library/functions.html#eval
.. _`serious security problems`: http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
