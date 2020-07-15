
Contexts
========

If you work frequently on certain topics, you will probably find the need to
convert between dimensions based on some pre-established (physical)
relationships. For example, in spectroscopy you need to transform from
wavelength to frequency. These are incompatible units and therefore Pint will
raise an error if you do this directly:

.. doctest::

    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> q = 500 * ureg.nm
    >>> q.to('Hz')
    Traceback (most recent call last):
    ...
    DimensionalityError: Cannot convert from 'nanometer' ([length]) to 'hertz' (1 / [time])


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
operations with the registry

.. doctest::

    >>> ureg.enable_contexts('sp')

To disable the context, just call

.. doctest::

    >>> ureg.disable_contexts()


Enabling multiple contexts
--------------------------

You can enable multiple contexts:

.. doctest::

    >>> q.to('Hz', 'sp', 'boltzmann')
    <Quantity(5.99584916e+14, 'hertz')>

This works also using the `with` statement:

.. doctest::

    >>> with ureg.context('sp', 'boltzmann'):
    ...     q.to('Hz')
    <Quantity(5.99584916e+14, 'hertz')>

or in the registry:

.. doctest::

    >>> ureg.enable_contexts('sp', 'boltzmann')
    >>> q.to('Hz')
    <Quantity(5.99584916e+14, 'hertz')>

If a conversion rule between two dimensions appears in more than one context,
the one in the last context has precedence. This is easy to remember if you
think that the previous syntax is equivalent to nest contexts:

.. doctest::

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
    <Quantity(398.4962..., 'nanometer')>

Contexts can also accept Pint Quantity objects as parameters. For example, the
'chemistry' context accepts the molecular weight of a substance (as a Quantity
with dimensions of [mass]/[substance]) to allow conversion between moles and
mass.

.. doctest::

    >>> substance = 95 * ureg('g')
    >>> substance.to('moles', 'chemistry', mw = 5 * ureg('g/mol'))
    <Quantity(19.0, 'mole')>


Ensuring context when calling a function
----------------------------------------

Pint provides a decorator to make sure that a function called is done within a given
context. Just like before, you have to provide as argument the name (or alias) of the
context and the parameters that you wish to set.


.. doctest::

    >>> wl = 530. * ureg.nm
    >>> @ureg.with_context('sp', n=1.33)
    ... def f(wl):
    ...     return wl.to('Hz').magnitude
    >>> f(wl)
    425297855014895.6


This decorator can be combined with **wraps** or **check** decorators described in
:doc:`wrapping`.


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
    ...                      lambda ureg, x: x / ureg.speed_of_light)
    >>> c.add_transformation('[time]', '[length]',
    ...                      lambda ureg, x: x * ureg.speed_of_light)
    >>> ureg.add_context(c)
    >>> ureg("1 s").to("km", "ab")
    <Quantity(299792.458, 'kilometer')>

It is also possible to create anonymous contexts without invoking add_context:

.. doctest::

   >>> c = pint.Context()
   >>> c.add_transformation('[time]', '[length]', lambda ureg, x: x * ureg.speed_of_light)
   >>> ureg("1 s").to("km", c)
   <Quantity(299792.458, 'kilometer')>

Using contexts for unit redefinition
------------------------------------

The exact definition of a unit of measure can change slightly depending on the country,
year, and more in general convention. For example, the ISO board released over the years
several revisions of its whitepapers, which subtly change the value of some of the more
obscure units. And as soon as one steps out of the SI system and starts wandering into
imperial and colonial measuring systems, the same unit may start being defined slightly
differently every time - with no clear 'right' or 'wrong' definition.

The default pint definitions file (default_en.txt) tries to mitigate the problem by
offering multiple variants of the same unit by calling them with different names; for
example, one will find multiple definitions of a "BTU"::

    british_thermal_unit = 1055.056 * joule = Btu = BTU = Btu_iso
    international_british_thermal_unit = 1e3 * pound / kilogram * degR / kelvin * international_calorie = Btu_it
    thermochemical_british_thermal_unit = 1e3 * pound / kilogram * degR / kelvin * calorie = Btu_th

That's sometimes insufficient, as Wikipedia reports `no less than 6 different
definitions <https://en.wikipedia.org/wiki/British_thermal_unit>`_ for BTU, and it's
entirely possible that some companies in the energy sector, or even individual energy
contracts, may redefine it to something new entirely, e.g. with a different rounding.

Pint allows changing the definition of a unit within the scope of a context.
This allows layering; in the example above, a company may use the global definition
of BTU from default_en.txt above, then override it with a customer-specific one in
a context, and then override it again with a contract-specific one on top of it.

A redefinition follows the following syntax::

    <unit name> = <new definition>

where <unit name> can be the base unit name or one of its aliases.
For example::

    BTU = 1055 J


Programmatically:

.. code-block:: python

    >>> ureg = pint.UnitRegistry()
    >>> q = ureg.Quantity("1 BTU")
    >>> q.to("J")
    1055.056 joule
    >>> ctx = pint.Context()
    >>> ctx.redefine("BTU = 1055 J")
    >>> q.to("J", ctx)
    1055.0 joule
    # When the context is disabled, pint reverts to the base definition
    >>> q.to("J")
    1055.056 joule

Or with a definitions file::

    @context somecontract
        BTU = 1055 J
    @end

.. code-block:: python

    >>> ureg = pint.UnitRegistry()
    >>> ureg.load_definitions("somefile.txt")
    >>> q = ureg.Quantity("1 BTU")
    >>> q.to("J")
    1055.056 joule
    >>> q.to("J", "somecontract")
    1055.0 joule


.. note::
   Redefinitions are transitive; if the registry defines B as a function of A
   and C as a function of B, redefining B will also impact the conversion from C to A.

**Limitations**

- You can't create brand new units ; all units must be defined outside of the context
  first.
- You can't change the dimensionality of a unit within a context. For example, you
  can't define a context that redefines grams as a force instead of a mass (but see
  the unit ``force_gram`` in default_en.txt).
- You can't redefine a unit with a prefix; e.g. you can redefine a liter, but not a
  decaliter.
- You can't redefine a base unit, such as grams.
- You can't add or remove aliases, or change the symbol. Symbol and aliases are
  automatically inherited from the UnitRegistry.
- You can't redefine dimensions or prefixes.

Working without a default definition
------------------------------------

In some cases, the definition of a certain unit may be so volatile to make it unwise to
define a default conversion rate in the UnitRegistry.

This can be solved by using 'NaN' (any capitalization) instead of a conversion rate rate
in the UnitRegistry, and then override it in contexts::

    truckload = nan kg

    @context Euro_TIR
        truckload = 2000 kg
    @end

    @context British_grocer
        truckload = 500 lb
    @end

This allows you, before any context is activated, to define quantities and perform
dimensional analysis:

.. code-block:: python

    >>> ureg.truckload.dimensionality
    [mass]
    >>> q = ureg.Quantity("2 truckloads")
    >>> q.to("kg")
    nan kg
    >>> q.to("kg", "Euro_TIR")
    4000 kilogram
    >>> q.to("kg", "British_grocer")
    453.59237 kilogram
