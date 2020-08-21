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
    >>> Q_ = ureg.Quantity
    >>> def mypp_caveman(length):
    ...     return pendulum_period(length.to(ureg.meter).magnitude) * ureg.second

and:

.. doctest::

    >>> mypp_caveman(100 * ureg.centimeter)
    <Quantity(2.00640929, 'second')>

Pint provides a more convenient way to do this:

.. doctest::

    >>> mypp = ureg.wraps(ureg.second, ureg.meter)(pendulum_period)

Or in the decorator format:

.. doctest::

    >>> @ureg.wraps(ureg.second, ureg.meter)
    ... def mypp(length):
    ...     return pendulum_period(length)
    >>> mypp(100 * ureg.centimeter)
    <Quantity(2.00640929, 'second')>


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
    <Quantity(2.00640929, 'second')>
    >>> mypp(1.)
    Traceback (most recent call last):
    ...
    ValueError: A wrapped function using strict=True requires quantity for all arguments with not None units. (error found for meter, 1.0)

To enable using non-Quantity numerical values, set strict to False`.

.. doctest::

    >>> mypp_ns = ureg.wraps(ureg.second, ureg.meter, False)(pendulum_period)
    >>> mypp_ns(1. * ureg.meter)
    <Quantity(2.00640929, 'second')>
    >>> mypp_ns(1.)
    <Quantity(2.00640929, 'second')>

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
airmass

.. doctest::

    @ureg.wraps(('deg', 'deg'), ('deg', 'deg', 'millibar', 'degC'))
    def solar_position(lat, lon, press, tamb, timestamp):
        return zenith, azimuth, airmass

Optional arguments
------------------

For a function with named keywords with optional values, use a tuple for all
arguments:

.. doctest::

    >>> @ureg.wraps(ureg.second, (ureg.meters, ureg.meters/ureg.second**2, None))
    ... def calculate_time_to_fall(height, gravity=Q_(9.8, 'm/s^2'), verbose=False):
    ...     """Calculate time to fall from a height h.
    ...
    ...     By default, the gravity is assumed to be earth gravity,
    ...     but it can be modified.
    ...
    ...     d = .5 * g * t**2
    ...     t = sqrt(2 * d / g)
    ...     """
    ...     t = math.sqrt(2 * height / gravity)
    ...     if verbose: print(str(t) + " seconds to fall")
    ...     return t
    ...
    >>> lunar_module_height = Q_(22, 'feet') + Q_(11, 'inches')
    >>> calculate_time_to_fall(lunar_module_height, verbose=True)
    1.1939473204801092 seconds to fall
    <Quantity(1.19394732, 'second')>
    >>> moon_gravity = Q_(1.625, 'm/s^2')
    >>> calculate_time_to_fall(lunar_module_height, gravity=moon_gravity)
    <Quantity(2.932051, 'second')>


Specifying relations between arguments
--------------------------------------

In certain cases, you may not be concerned with the actual units and only care about the unit relations among arguments.

This is done using a string starting with the equal sign `=`:

.. doctest::

    >>> @ureg.wraps('=A**2', ('=A', '=A'))
    ... def sqsum(x, y):
    ...     return x * x  + 2 * x * y + y * y

which can be read as the first argument (`x`) has certain units (we labeled them `A`),
the second argument (`y`) has the same units as the first (`A` again). The return value
has the unit of `x` squared (`A**2`)

You can use more than one label:

.. doctest::

    >>> @ureg.wraps('=A**2*B', ('=A', '=A*B', '=B'))
    ... def some_function(x, y, z):
    ...     pass

With optional arguments

.. doctest::

    >>> @ureg.wraps('=A*B', ('=A', '=B'))
    ... def get_displacement(time, rate=Q_(1, 'm/s')):
    ...     return time * rate
    ...
    >>> get_displacement(Q_(2, 's'))
    <Quantity(2, 'meter')>
    >>> get_displacement(Q_(2, 's'), Q_(1, 'deg/s'))
    <Quantity(2, 'degree')>


Ignoring an argument or return value
------------------------------------

To avoid the conversion of an argument or return value, use None

.. doctest::

    >>> mypp3 = ureg.wraps((ureg.second, None), ureg.meter)(pendulum_period_error)


Checking dimensionality
=======================

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

If you just want to check the dimensionality of a quantity, you can do so with the built-in 'check' function.

.. doctest::

    >>> distance = 1 * ureg.m
    >>> distance.check('[length]')
    True
    >>> distance.check('[time]')
    False
