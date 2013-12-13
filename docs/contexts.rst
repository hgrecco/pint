.. _contexts:

Contexts
========

If you work frequently on certain topics, you will probably find the need
to convert between units based some pre-established (physical) relationships.
For example, if you are working in spectroscopy you need to transform
wavelength to frequency. These are incompatible units and therefore Pint
will raise an error if your do this directly:

.. doctest::

    >>> ureg = pint.UnitRegistry()
    >>> q = 500 * ureg.nm
    >>> q.to('Hz')
    Traceback (most recent call last):
    ...
    pint.unit.DimensionalityError: Cannot convert from 'nanometer' ([length]) to 'hertz' (1 / [time])


You need to use the relation `wavelength = speed_of_light * frequency`:

.. doctest::

    >>> (ureg.speed_of_light / q).to('Hz')
    <Quantity(5.99584916e+14, 'hertz')>


To make this task easy, Pint has the concept of `contexts` which provides convertion
rules between dimensions. For example, the relation between wavelength and frequency is
defined in the spectroscopy context. You can tell pint to use this context when you
convert a quantity to different units.

..doctest::

    >>> q.to('Hz', context='spectroscopy')
    <Quantity(5.99584916e+14, 'hertz')>

or with the abbreviated form:

..doctest::

    >>> q.to('Hz', context='sp')
    <Quantity(5.99584916e+14, 'hertz')>

or even shorter:

..doctest::

    >>> q.to('Hz', 'sp')
    <Quantity(5.99584916e+14, 'hertz')>


Contexts can be also enabled for blocks of code using the `with` statement:

..doctest::

    >>> with ureg.context('sp'):
    >>>     q.to('Hz')
    <Quantity(5.99584916e+14, 'hertz')>

If you need a particular context in all your code, you can enable it for all
operations with the registry::

    >>> ureg.enable_contexts('sp')

To disable the context, just call::

    >>> ureg.disable_contexts()


Enabling multiple contexts
--------------------------

Using a tuple of contexts names you can enable multiple contexts:

    >>> q.to('Hz', context=('sp', 'boltzmann'))

This works also using the `with` statement:

    >>> with ureg.context('sp', 'boltzmann'):
    ...     q.to('Hz')

or in the registry:

    >>> ureg.enable_contexts('sp', 'boltzmann')
    >>> q.to('Hz')

If a conversion rule between two dimensions appears in more than one context,
those in the last context have precedence. This is easy to remember if you think
that the previous syntax is equivalent to nest contexts:

    >>> with ureg.context('sp'):
    ...     with ureg.context('boltzmann') :
    ...         q.to('Hz')


Parameterized contexts
----------------------

Contexts can also take named parameters. For example, in the spectroscopy you
can specify the index of refraction of the medium (`n`). In this way you can
calculate, for example, the wavelength in water of a laser which on air is 530 nm.

..doctest::

    >>> wl = 530. * ureg.nm
    >>> f = wl.to('Hz', 'sp')
    >>> f.to('wl', 'sp', n=1.33)




Defining contexts in a file
---------------------------

Like all units and dimensions in Pint, `contexts` are defined using an easy to
understand text syntax. For example, the definition of the spectroscopy
context is::

    @context(n=1) spectroscopy = sp
        # n index of refraction of the medium.
        [length] <-> [frequency]: speed_of_light / n / value
        [frequency] -> [energy]: planck_constant * value
        [energy] -> [frequency]: value / planck_constant
    @end

The `@context` directive indicates the beginning of the definitions which are finished by an
`@end` statement. You can optionally specify parameters for the context in parenthesis.
All parameters are named and default values are mandatory. Multiple parameters
are separated by commas (like in a python function definition). Finally you provide the name
of the context and, optionally, a short version of the name separated by an equal sign.

Conversions rules are specified by providing source and destination dimensions separated
using a colon (`:`) from the equation. A special variable named `value` will be replaced
by the source quantity. Other names will be looked first in the context arguments and
then in registry.

A single forward arrow (`->`) indicates that the equations is used to transform
from the first dimension to the second one. A double arrow (`<->`) is used to
indicate that the transformation operates both ways.


Defining contexts programmatically
----------------------------------

You can create `Context` object, and populate the conversion rules using python functions.
For example:

..doctest::

    >>> ureg = pint.UnitRegistry()
    >>> c = pint.Context('ab')
    >>> c.add_transformation('[length]', '[time]', lambda ureg, x: ureg.speed_of_light / x)
    >>> c.add_transformation('[time]', '[length]', lambda ureg, x: ureg.speed_of_light * x)
    >>> ureg.add_context(c)
