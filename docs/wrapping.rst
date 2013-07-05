.. _wrapping:

Wrapping functions
==================

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

To understand the syntax, consider the usage in the decorator format:

.. doctest::

    >>> @ureg.wraps(ureg.second, ureg.meter)
    ... def mypp(length):
    ...     return pendulum_period(length)

`wraps` takes 3 input arguments::

    - **ret**: the return units.
               Use None to skip conversion.
    - **args**: the inputs units for each argument, as an iterable.
                Use None to skip conversion of any given element.
    - **strict**: if `True` all convertible arguments must be a Quantity
                  and others will raise a ValueError (True by default)

    >>> mypp(100 * ureg.centimeter)
    <Quantity(2.0064092925890407, 'second')>

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

Ignoring an argument or return value
------------------------------------

To avoid the conversion of an argument or return value, use None

.. doctest::

    >>> mypp3 = ureg.wraps((ureg.second, None), ureg.meter)(pendulum_period_error)




