.. _nonmult:


Temperature conversion
======================

Unlike meters and seconds, fahrenheits, celsius and kelvin are not
multiplicative units. Temperature is expressed in a system with a
reference point, and relations between temperature units include
not only an scaling factor but also an offset. Pint supports these
type of units and conversions between them. The default definition
file includes fahrenheits, celsius, kelvin and rankine abbreviated
as degF, degC, degK, and degR.

For example, to convert from celsius to fahrenheit:

.. testsetup:: *

   from pint import UnitRegistry
   ureg = UnitRegistry()
   Q_ = ureg.Quantity

.. doctest::

   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry()
   >>> home = 25.4 * ureg.degC
   >>> print(home.to('degF'))
   77.72000039999993 degF

or to other kelvin or rankine:

.. doctest::

    >>> print(home.to('degK'))
    298.54999999999995 degK
    >>> print(home.to('degR'))
    537.39 degR

Additionally, for every temperature unit in the registry,
there is also a *delta* counterpart to specify differences.
For example, the change in celsius is equal to the change
in kelvin, but not in fahrenheit (as the scaling factor
is different).

.. doctest::

   >>> increase = 12.3 * ureg.delta_degC
   >>> print(increase.to(ureg.delta_degK))
   12.3 delta_degK
   >>> print(increase.to(ureg.delta_degF))
   6.833333333333334 delta_degF

..
    Subtraction of two temperatures also yields a *delta* unit.

    .. doctest::

        >>> 25.4 * ureg.degC - 10. * ureg.degC
        15.4 delta_degC

Differences in temperature are multiplicative:

.. doctest::

    >>> speed = 60. * ureg.delta_degC / ureg.min
    >>> print(speed.to('delta_degC/second'))
    1.0 delta_degC / second

The parser knows about *delta* units and use them when a temperature unit
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

    >>> print(ureg.parse_units('degC/meter', to_delta=False))
    degC / meter


To define a new temperature, you need to specify the offset. For example,
this is the definition of the celsius and fahrenheit::

    degC = degK; offset: 273.15 = celsius
    degF = 9 / 5 * degK; offset: 255.372222 = fahrenheit

You do not need to define *delta* units, as they are defined automatically.

