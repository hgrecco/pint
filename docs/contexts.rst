.. _contexts:

Contexts
========

If you work frequently on certain topics, you will probably find the need to
convert between dimensions based on some pre-established (physical)
relationships. For example, in spectroscopy you need to transform from
wavelength to frequency. These are incompatible units and therefore Pint will
raise an error if your do this directly:

.. doctest::

    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> q = 500 * ureg.nm
    >>> q.to('Hz')
    Traceback (most recent call last):
    ...
    pint.errors.DimensionalityError: Cannot convert from 'nanometer' ([length]) to 'hertz' (1 / [time])


You probably want to use the relation `frequency = speed_of_light / wavelength`:

.. doctest::

    >>> (ureg.speed_of_light / q).to('Hz')
    <Quantity(5.99584916e+14, 'hertz')>


To make this task easy, Pint has the concept of `contexts` which provides
conversion rules between dimensions. For example, the relation between
wavelength and frequency is defined in the `spectroscopy` context (abbreviated
`sp`). You can tell pint to use this context when you convert a quantity to
different units.

.. doctest::

    >>> q.to('Hz', 'spectroscopy')
    <Quantity(5.99584916e+14, 'hertz')>

or with the abbreviated form:

.. doctest::

    >>> q.to('Hz', 'sp')
    <Quantity(5.99584916e+14, 'hertz')>

Contexts can be also enabled for blocks of code using the `with` statement:

.. doctest::

    >>> with ureg.context('sp'):
    ...     q.to('Hz')
    <Quantity(5.99584916e+14, 'hertz')>

If you need a particular context in all your code, you can enable it for all
operations with the registry::

    >>> ureg.enable_contexts('sp')

To disable the context, just call::

    >>> ureg.disable_contexts()


Enabling multiple contexts
--------------------------

You can enable multiple contexts:

    >>> q.to('Hz', 'sp', 'boltzmann')
    <Quantity(5.99584916e+14, 'hertz')>

This works also using the `with` statement:

    >>> with ureg.context('sp', 'boltzmann'):
    ...     q.to('Hz')
    <Quantity(5.99584916e+14, 'hertz')>

or in the registry:

    >>> ureg.enable_contexts('sp', 'boltzmann')
    >>> q.to('Hz')
    <Quantity(5.99584916e+14, 'hertz')>

If a conversion rule between two dimensions appears in more than one context,
the one in the last context has precedence. This is easy to remember if you
think that the previous syntax is equivalent to nest contexts:

    >>> with ureg.context('sp'):
    ...     with ureg.context('boltzmann') :
    ...         q.to('Hz')
    <Quantity(5.99584916e+14, 'hertz')>


Parameterized contexts
----------------------

Contexts can also take named parameters. For example, in the spectroscopy you
can specify the index of refraction of the medium (`n`). In this way you can
calculate, for example, the wavelength in water of a laser which on air is 530 nm.

.. doctest::

    >>> wl = 530. * ureg.nm
    >>> f = wl.to('Hz', 'sp')
    >>> f.to('nm', 'sp', n=1.33)
    <Quantity(398.496240602, 'nanometer')>

Contexts can also accept Pint Quantity objects as parameters. For example, the
'chemistry' context accepts the molecular weight of a substance (as a Quantity
with dimensions of [mass]/[substance]) to allow conversion between moles and
mass.

.. doctest::

    >>> substance = 95 * ureg('g')
    >>> substance.to('moles', 'chemistry', mw = 5 * ureg('g/mol'))
    <Quantity(19.0, 'mole')>


Defining contexts in a file
---------------------------

Like all units and dimensions in Pint, `contexts` are defined using an easy to
read text syntax. For example, the definition of the spectroscopy
context is::

    @context(n=1) spectroscopy = sp
        # n index of refraction of the medium.
        [length] <-> [frequency]: speed_of_light / n / value
        [frequency] -> [energy]: planck_constant * value
        [energy] -> [frequency]: value / planck_constant
    @end

The `@context` directive indicates the beginning of the transformations which
are finished by the `@end` statement. You can optionally specify parameters for
the context in parenthesis. All parameters are named and default values are
mandatory. Multiple parameters are separated by commas (like in a python
function definition). Finally, you provide the name of the context (e.g.
spectroscopy) and, optionally, a short version of the name (e.g. sp) separated
by an equal sign. See the definition of the 'chemistry' context in
default_en.txt for an example of a multiple-parameter context.

Conversions rules are specified by providing source and destination dimensions
separated using a colon (`:`) from the equation. A special variable named
`value` will be replaced by the source quantity. Other names will be looked
first in the context arguments and then in registry.

A single forward arrow (`->`) indicates that the equations is used to transform
from the first dimension to the second one. A double arrow (`<->`) is used to
indicate that the transformation operates both ways.

Context definitions are stored and imported exactly like custom units
definition file (and can be included in the same file as unit definitions). See
"Defining units" for details.

Defining contexts programmatically
----------------------------------

You can create `Context` object, and populate the conversion rules using python
functions. For example:

.. doctest::

    >>> ureg = pint.UnitRegistry()
    >>> c = pint.Context('ab')
    >>> c.add_transformation('[length]', '[time]',
    ...                      lambda ureg, x: ureg.speed_of_light / x)
    >>> c.add_transformation('[time]', '[length]',
    ...                      lambda ureg, x: ureg.speed_of_light * x)
    >>> ureg.add_context(c)
