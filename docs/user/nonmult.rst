.. _nonmult:


Temperature conversion
======================

Unlike meters and seconds, the temperature units fahrenheits and
celsius are non-multiplicative units. These temperature units are
expressed in a system with a reference point, and relations between
temperature units include not only a scaling factor but also an offset.
Pint supports these type of units and conversions between them.
The default definition file includes fahrenheits, celsius,
kelvin and rankine abbreviated as degF, degC, degK, and degR.

For example, to convert from celsius to fahrenheit:

.. doctest::

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
    >>> ureg.default_format = '.3f'
    >>> Q_ = ureg.Quantity
    >>> home = Q_(25.4, ureg.degC)
    >>> print(home.to('degF'))
    77.720 degree_Fahrenheit

or to other kelvin or rankine:

.. doctest::

    >>> print(home.to('kelvin'))
    298.550 kelvin
    >>> print(home.to('degR'))
    537.390 degree_Rankine

Additionally, for every non-multiplicative temperature unit
in the registry, there is also a *delta* counterpart to specify
differences. Absolute units have no *delta* counterpart.
For example, the change in celsius is equal to the change
in kelvin, but not in fahrenheit (as the scaling factor
is different).

.. doctest::

    >>> increase = 12.3 * ureg.delta_degC
    >>> print(increase.to(ureg.kelvin))
    12.300 kelvin
    >>> print(increase.to(ureg.delta_degF))
    22.140 delta_degree_Fahrenheit

Subtraction of two temperatures given in offset units yields a *delta* unit:

.. doctest::

    >>> Q_(25.4, ureg.degC) - Q_(10., ureg.degC)
    <Quantity(15.4, 'delta_degree_Celsius')>

You can add or subtract a quantity with *delta* unit and a quantity with
offset unit:

.. doctest::

    >>> Q_(25.4, ureg.degC) + Q_(10., ureg.delta_degC)
    <Quantity(35.4, 'degree_Celsius')>
    >>> Q_(25.4, ureg.degC) - Q_(10., ureg.delta_degC)
    <Quantity(15.4, 'degree_Celsius')>

If you want to add a quantity with absolute unit to one with offset unit, like here

.. doctest::

    >>> heating_rate = 0.5 * ureg.kelvin/ureg.min
    >>> Q_(10., ureg.degC) + heating_rate * Q_(30, ureg.min)
    Traceback (most recent call last):
            ...
    OffsetUnitCalculusError: Ambiguous operation with offset unit (degC, kelvin).

you have to avoid the ambiguity by either converting the offset unit to the
absolute unit before addition

.. doctest::

    >>> Q_(10., ureg.degC).to(ureg.kelvin) + heating_rate * Q_(30, ureg.min)
    <Quantity(298.15, 'kelvin')>

or convert the absolute unit to a *delta* unit:

.. doctest::

    >>> Q_(10., ureg.degC) + heating_rate.to('delta_degC/min') * Q_(30, ureg.min)
    <Quantity(25.0, 'degree_Celsius')>

In contrast to subtraction, the addition of quantities with offset units
is ambiguous, e.g. for *10 degC + 100 degC* two different result are reasonable
depending on the context, *110 degC* or *383.15 °C (= 283.15 K + 373.15 K)*.
Because of this ambiguity pint raises an error for the addition of two
quantities with offset units (since pint-0.6).

Quantities with *delta* units are multiplicative:

.. doctest::

    >>> speed = 60. * ureg.delta_degC / ureg.min
    >>> print(speed.to('delta_degC/second'))
    1.000 delta_degree_Celsius / second

However, multiplication, division and exponentiation of quantities with
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
    <Quantity(25.4, 'degree_Celsius')>

As an alternative to raising an error, pint can be configured to work more
relaxed via setting the UnitRegistry parameter *autoconvert_offset_to_baseunit*
to true. In this mode, pint behaves differently:

* Multiplication of a quantity with a single offset unit with order +1 by
  a number or ndarray yields the quantity in the given unit.

.. doctest::

    >>> ureg = UnitRegistry(autoconvert_offset_to_baseunit = True)
    >>> T = 25.4 * ureg.degC
    >>> T
    <Quantity(25.4, 'degree_Celsius')>

* Before all other multiplications, all divisions and in case of
  exponentiation [#f1]_ involving quantities with offset-units, pint
  will convert the quantities with offset units automatically to the
  corresponding base unit before performing the operation.

.. doctest::

    >>> 1/T
    <Quantity(0.0033495..., '1 / kelvin')>
    >>> T * 10 * ureg.meter
    <Quantity(527.15, 'kelvin * meter')>

You can change the behaviour at any time:

.. doctest::

    >>> ureg.autoconvert_offset_to_baseunit = False
    >>> 1/T
    Traceback (most recent call last):
        ...
    OffsetUnitCalculusError: Ambiguous operation with offset unit (degC).

The parser knows about *delta* units and uses them when a temperature unit
is found in a multiplicative context. For example, here:

.. doctest::

    >>> print(ureg.parse_units('degC/meter'))
    delta_degree_Celsius / meter

but not here:

.. doctest::

    >>> print(ureg.parse_units('degC'))
    degree_Celsius

You can override this behaviour:

.. doctest::

    >>> print(ureg.parse_units('degC/meter', as_delta=False))
    degree_Celsius / meter

Note that the magnitude is left unchanged:

.. doctest::

    >>> Q_(10, 'degC/meter')
    <Quantity(10, 'delta_degree_Celsius / meter')>

To define a new temperature, you need to specify the offset. For example,
this is the definition of the celsius and fahrenheit::

    degC = degK; offset: 273.15 = celsius
    degF = 5 / 9 * degK; offset: 255.372222 = fahrenheit

You do not need to define *delta* units, as they are defined automatically.

.. [#f1] If the exponent is +1, the quantity will not be converted to base
         unit but remains unchanged.
