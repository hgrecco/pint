.. _wrapping:

Wrapping and checking functions
===============================

In some cases you might want to use pint with a pre-existing web service or library
which is not units aware. Or you might want to write a fast implementation of a
numerical algorithm that requires the input values in some specific units.

For example, consider a function to return the period of the pendulum within
a hypothetical physics library. The library does not use units, but instead
requires you to provide numerical values in certain units:

.. testsetup:: *

   import math
   G = 9.806650
   def pendulum_period(length):
       return 2*math.pi*math.sqrt(length/G)

   def pendulum_period2(length, swing_amplitude):
       pass

   def pendulum_period_maxspeed(length, swing_amplitude):
       pass

   def pendulum_period_error(length):
       pass

.. doctest::

    >>> from simple_physics import pendulum_period      # doctest: +SKIP
    >>> help(pendulum_period)                           # doctest: +SKIP
    Help on function pendulum_period in module simple_physics:

    pendulum_period(length)
    Return the pendulum period in seconds. The length of the pendulum
    must be provided in meters.

    >>> pendulum_period(1)
    2.0064092925890407

This behaviour is very error prone, in particular when combining multiple libraries.
You could wrap this function to use Quantities instead:

.. doctest::

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
    >>> def mypp_caveman(length):
    ...     return pendulum_period(length.to(ureg.meter).magnitude) * ureg.second

and:

.. doctest::

    >>> mypp_caveman(100 * ureg.centimeter)
    <Quantity(2.0064092925890407, 'second')>

Pint provides a more convenient way to do this:

.. doctest::

    >>> mypp = ureg.wraps(ureg.second, ureg.meter)(pendulum_period)

Or in the decorator format:

.. doctest::

    >>> @ureg.wraps(ureg.second, ureg.meter)
    ... def mypp(length):
    ...     return pendulum_period(length)
    >>> mypp(100 * ureg.centimeter)
    <Quantity(2.0064092925890407, 'second')>


`wraps` takes 3 input arguments:

    - **ret**: the return units.
               Use None to skip conversion.
    - **args**: the inputs units for each argument, as an iterable.
                Use None to skip conversion of any given element.
    - **strict**: if `True` all convertible arguments must be a Quantity
                  and others will raise a ValueError (True by default)



Strict Mode
-----------

By default, the function is wrapped in `strict` mode. In this mode,
the input arguments assigned to units must be a Quantities.

.. doctest::

    >>> mypp(1. * ureg.meter)
    <Quantity(2.0064092925890407, 'second')>
    >>> mypp(1.)
    Traceback (most recent call last):
    ...
    ValueError: A wrapped function using strict=True requires quantity for all arguments with not None units. (error found for meter, 1.0)

To enable using non-Quantity numerical values, set strict to False`.

.. doctest::

    >>> mypp_ns = ureg.wraps(ureg.second, ureg.meter, False)(pendulum_period)
    >>> mypp_ns(1. * ureg.meter)
    <Quantity(2.0064092925890407, 'second')>
    >>> mypp_ns(1.)
    <Quantity(2.0064092925890407, 'second')>

In this mode, the value is assumed to have the correct units.


Multiple arguments or return values
-----------------------------------

For a function with more arguments, use a tuple:

.. doctest::

    >>> from simple_physics import pendulum_period2         # doctest: +SKIP
    >>> help(pendulum_period2)                              # doctest: +SKIP
    Help on function pendulum_period2 in module simple_physics:

    pendulum_period2(length, swing_amplitude)
    Return the pendulum period in seconds. The length of the pendulum
    must be provided in meters. The swing_amplitude must be in radians.

    >>> mypp2 = ureg.wraps(ureg.second, (ureg.meter, ureg.radians))(pendulum_period2)
    ...

Or if the function has multiple outputs:

.. doctest::

    >>> mypp3 = ureg.wraps((ureg.second, ureg.meter / ureg.second),
    ...                    (ureg.meter, ureg.radians))(pendulum_period_maxspeed)
    ...

If there are more return values than specified units, ``None`` is assumed for
the extra outputs. For example, given the NREL SOLPOS calculator that outputs
solar zenith, azimuth and air mass, the following wrapper assumes no units for
airmass::

    @UREG.wraps(('deg', 'deg'), ('deg', 'deg', 'millibar', 'degC')
    def solar_position(lat, lon, press, tamb, timestamp):
        return zenith, azimuth, airmass


Specifying relations between arguments
--------------------------------------

In certain cases the actual units but just their relation. This is done using string
starting with the equal sign `=`:

.. doctest::

    >>> @ureg.wraps('=A**2', ('=A', '=A'))
    ... def sqsum(x, y):
    ...     return x * x  + 2 * x * y + y * y

which can be read as the first argument (`x`) has certain units (we labeled them `A`),
the second argument (`y`) has the same units as the first (`A` again). The return value
has the unit of `x` squared (`A**2`)

You can use more than one label:

    >>> @ureg.wraps('=A**2*B', ('=A', '=A*B', '=B'))
    ... def some_function(x, y, z):
    ...     pass


Ignoring an argument or return value
------------------------------------

To avoid the conversion of an argument or return value, use None

.. doctest::

    >>> mypp3 = ureg.wraps((ureg.second, None), ureg.meter)(pendulum_period_error)


Checking units
==============

When you want pint quantities to be used as inputs to your functions, pint provides a wrapper to ensure units are of
correct type - or more precisely, they match the expected dimensionality of the physical quantity.

Similar to wraps(), you can pass None to skip checking of some parameters, but the return parameter type is not checked.

.. doctest::

    >>> mypp = ureg.check('[length]')(pendulum_period)

In the decorator format:

.. doctest::

    >>> @ureg.check('[length]')
    ... def pendulum_period(length):
    ...     return 2*math.pi*math.sqrt(length/G)

