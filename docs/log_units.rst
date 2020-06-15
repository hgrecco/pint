.. _log_units:


Logarithmic Units
=================

Pint supports some logarithmic units, including dB, dBm, octave, and decade
as well as conversions between them and their base units where applicable.
These units behave much like those described in :ref:`nonmult`, so many of
the recommendations there apply here as well.

.. note::

    If you're making heavy use of logarithmic units, you may find it helpful to
    pass ``autoconvert_offset_to_baseunit=True`` when initializing your ``UnitRegistry()``,
    this will allow you to use syntax like ``10.0 * ureg.dBm`` in lieu of the
    explicit ``Quantity()`` constructor. Many examples on this page assume
    you've passed this parameter, and will not work otherwise.

.. testsetup::

   from pint import UnitRegistry
   ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
   ureg.default_format = '.3f'
   Q_ = ureg.Quantity

.. doctest::

   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
   >>> Q_ = ureg.Quantity
   >>> signal_power = -20.0 * ureg.dBm
   >>> print(f"{sp.to('milliwatts'):0.3~P}")
   0.01 mW

TODO:
[ ] explain delta units (what purpose do they have here?)
[ ] example computing mW to dBm
[ ] example computing dB and (something)


Multiplication, division and exponentiation of quantities with
offset units is problematic just like addition. Pint (since version 0.6)
will by default raise an error when a quantity with offset unit is used in
these operations. Due to this quantities with offset units cannot be created
like other quantities by multiplication of magnitude and unit but have
to be explicitly created:

.. doctest::

    >>> ureg = UnitRegistry()
    >>> home = 25.4 * ureg.degC
    Traceback (most recent call last):
        ...
    OffsetUnitCalculusError: Ambiguous operation with offset unit (degC).
    >>> Q_(25.4, ureg.degC)
    <Quantity(25.4, 'degC')>

As an alternative to raising an error, pint can be configured to work more
relaxed via setting the UnitRegistry parameter *autoconvert_offset_to_baseunit*
to true. In this mode, pint behaves differently:

* Multiplication of a quantity with a single offset unit with order +1 by
  a number or ndarray yields the quantity in the given unit.

.. doctest::

    >>> ureg = UnitRegistry(autoconvert_offset_to_baseunit = True)
    >>> T = 25.4 * ureg.degC
    >>> T
    <Quantity(25.4, 'degC')>

* Before all other multiplications, all divisions and in case of
  exponentiation [#f1]_ involving quantities with offset-units, pint
  will convert the quantities with offset units automatically to the
  corresponding base unit before performing the operation.

    >>> 1/T
    <Quantity(0.00334952269302, '1 / kelvin')>
    >>> T * 10 * ureg.meter
    <Quantity(527.15, 'kelvin * meter')>

You can change the behaviour at any time:

    >>> ureg.autoconvert_offset_to_baseunit = False
    >>> 1/T
    Traceback (most recent call last):
        ...
    OffsetUnitCalculusError: Ambiguous operation with offset unit (degC).

The parser knows about *delta* units and uses them when a temperature unit
is found in a multiplicative context. For example, here:

.. doctest::

    >>> print(ureg.parse_units('degC/meter'))
    delta_degC / meter

but not here:

.. doctest::

    >>> print(ureg.parse_units('degC'))
    degC

You can override this behaviour:

.. doctest::

    >>> print(ureg.parse_units('degC/meter', as_delta=False))
    degC / meter

Note that the magnitude is left unchanged:

.. doctest::

    >>> Q_(10, 'degC/meter')
    <Quantity(10, 'delta_degC / meter')>

To define a new temperature, you need to specify the offset. For example,
this is the definition of the celsius and fahrenheit::

    degC = degK; offset: 273.15 = celsius
    degF = 5 / 9 * degK; offset: 255.372222 = fahrenheit

You do not need to define *delta* units, as they are defined automatically.

.. [#f1] If the exponent is +1, the quantity will not be converted to base
         unit but remains unchanged.
