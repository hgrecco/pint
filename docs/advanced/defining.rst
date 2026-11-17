.. _defining:

Defining units
==============


In a definition file
--------------------

To define units in a persistent way you need to create a unit definition file.
Such files are simple text files in which the units are defined as function of
other units. For example this is how the minute and the hour are defined in
`default_en.txt`::

    hour = 60 * minute = h = hr
    minute = 60 * second = min

It is quite straightforward, isn't it? We are saying that `minute` is
`60 seconds` and is also known as `min`.

1. The first word is always the canonical name.
2. Next comes the definition (based on other units).
3. Next, optionally, there is the unit symbol.
4. Finally, again optionally, a list of aliases, separated by equal signs.
   If one wants to specify aliases but not a symbol, the symbol should be
   conventionally set to ``_``; e.g.::

        millennium = 1e3 * year = _ = millennia

The order in which units are defined does not matter, Pint will resolve the
dependencies to define them in the right order. What is important is that if
you transverse all definitions, a reference unit is reached. A reference unit
is not defined as a function of another units but of a dimension. For the time
in `default_en.txt`, this is the `second`::

    second = [time] = s = sec

By defining `second` as equal to a string `time` in square brackets we indicate
that:

 * `time` is a physical dimension.
 * `second` is a reference unit.

The ability to define basic physical dimensions as well as reference units
allows to construct arbitrary units systems.

Pint is shipped with a default definition file named `default_en.txt` where
`en` stands for English. You can add your own definitions to the end of this
file but you will have to be careful to merge when you update Pint. An easier
way is to create a new file (e.g. `mydef.txt`) with your definitions::

   dog_year = 52 * day = dy

and then in Python, you can load it as:

   >>> from pint import UnitRegistry
   >>> # First we create the registry.
   >>> ureg = UnitRegistry()
   >>> # Then we append the new definitions
   >>> ureg.load_definitions('/your/path/to/my_def.txt') # doctest: +SKIP

If you make a translation of the default units or define a completely new set,
you don't want to append the translated definitions so you just give the
filename to the constructor::

   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry('/your/path/to/default_es.txt') # doctest: +SKIP

In the definition file, prefixes are identified by a trailing dash::

   yocto- = 10.0**-24 = y-

It is important to note that prefixed defined in this way can be used with any
unit, including non-metric ones (e.g. kiloinch is valid for Pint). This
simplifies definitions files enormously without introducing major problems.
Pint, like Python, believes that we are all consenting adults.

Derived dimensions are defined as follows::

    [density] = [mass] / [volume]

Note that primary dimensions don't need to be declared; they can be
defined for the first time as part of a unit definition.

Finally, one may add aliases to an already existing unit definition::

    @alias meter = metro = metr

This is particularly useful when one wants to enrich definitions from defaults_en.txt
with new aliases from a custom file. It can also be used for translations (like in the
example above) as long as one is happy to have the localized units automatically
converted to English when they are parsed.

Unit aliases must be nonempty single names. Whitespace, ASCII quotation marks,
numeric literals, and expression operators are not allowed. Unicode names,
including degree spellings such as ``°C``, and the standalone ``%`` and ``‰``
are supported.


Programmatically
----------------

You can easily add units, dimensions, or aliases to the registry programmatically.
Let's add a dog_year (sometimes written as dy) equivalent to 52 (human) days:

.. doctest::

   >>> from pint import UnitRegistry
   >>> # We first instantiate the registry.
   >>> # If we do not provide any parameter, the default unit definitions are used.
   >>> ureg = UnitRegistry()
   >>> Q_ = ureg.Quantity

   # Here we add the unit
   >>> ureg.define('dog_year = 52 * day = dy')

   # We create a quantity based on that unit and we convert to years.
   >>> lassie_lifespan = Q_(10, 'year')
   >>> print(lassie_lifespan.to('dog_years'))
   70.240384... dog_year

Note that we have used the name `dog_years` even though we have not defined the
plural form as an alias. Pint takes care of that, so you don't have to.
Plural forms that aren't simply built by adding a 's' suffix to the singular form
should be explicitly stated as aliases (see for example ``millennia`` above).

You can also add prefixes programmatically:

.. doctest::

   >>> ureg.define('myprefix- = 30 = my-')

where the number indicates the multiplication factor.

Same for aliases and derived dimensions:

.. doctest::

   >>> ureg.define('@alias meter = metro = metr')
   >>> ureg.define('[hypervolume] = [length] ** 4')


.. warning::
   Units, prefixes, aliases and dimensions added programmatically are forgotten when the
   program ends.


Redefining units
----------------

Defining a unit that already exists replaces it, and logs a warning unless the
registry was created with ``on_redefinition="ignore"`` (``"raise"`` raises a
:class:`~pint.RedefinitionError` instead). Conversions use the new definition
straight away, including for units derived from it:

.. doctest::

   >>> ureg2 = UnitRegistry(on_redefinition="ignore")
   >>> print(ureg2.Quantity(1, "lb").to("kg"))
   0.4535923... kilogram
   >>> ureg2.define("pound = 0.5 kg = lb")
   >>> print(ureg2.Quantity(1, "lb").to("kg"))
   0.5 kilogram
   >>> print(ureg2.Quantity(1, "ton").to("kg"))
   1000.0 kilogram

Each redefinition rebuilds the registry's cache. To redefine many units, pass
``rebuild_cache=False`` to ``define`` or ``load_definitions`` and the cache is
rebuilt once, by the next call without it:

.. doctest::

   >>> ureg2.define("pound = 0.4 kg = lb", rebuild_cache=False)
   >>> ureg2.define("ounce = 0.025 kg = oz")
   >>> print(ureg2.Quantity(1, "lb").to("kg"))
   0.4 kilogram

Until then, conversions may still use the old definitions.


Units with constants
--------------------

Some units, such as ``L/100km``, contain constants. These can be defined with a
leading underscore:

.. doctest::

   >>> ureg.define('_100km = 100 * kilometer')
   >>> ureg.define('mpg = 1 * mile / gallon')
   >>> fuel_ec_europe = 5 * ureg.L / ureg._100km
   >>> fuel_ec_us = (1 / fuel_ec_europe).to(ureg.mpg)


Units with a custom converter
-----------------------------

Definition strings can only express multiplicative, offset (see :ref:`nonmult`)
and logarithmic conversions. For any other invertible mapping, subclass
:class:`pint.converters.Converter` and register a
:class:`~pint.facets.plain.definitions.UnitDefinition` that uses it. These are
lower-level APIs than ``ureg.define('...')``.

For example, a road grade in percent is ``100 * tan(angle)``:

.. doctest::

   >>> import math
   >>> from dataclasses import dataclass
   >>> from pint.converters import Converter
   >>> from pint.facets.plain.definitions import UnitDefinition
   >>> from pint.util import UnitsContainer

   >>> @dataclass(frozen=True)
   ... class TangentConverter(Converter):
   ...     tan_scale: float
   ...
   ...     @property
   ...     def is_multiplicative(self):
   ...         return False
   ...
   ...     def to_reference(self, value, inplace=False):
   ...         return math.atan(value / self.tan_scale)
   ...
   ...     def from_reference(self, value, inplace=False):
   ...         return self.tan_scale * math.tan(value)

   >>> ureg.define(
   ...     UnitDefinition(
   ...         "grade_percent",  # name
   ...         "gradepct",  # symbol
   ...         (),  # aliases
   ...         TangentConverter(tan_scale=100),
   ...         UnitsContainer(radian=1),  # reference unit
   ...     )
   ... )
   >>> print(Q_(45, "degree").to("gradepct"))
   99.99999999... grade_percent
   >>> print(Q_(10, "gradepct").to("degree"))
   5.7105931374... degree

``to_reference`` converts a value in the new unit to the reference unit, and
``from_reference`` does the reverse. Use ``numpy`` functions instead of ``math``
to support array magnitudes.

As with offset units, arithmetic such as ``Q_(10, "gradepct") * 2`` raises an
error. Also, because ``radian`` is dimensionless, ``.to("percent")`` returns the
angle in radians expressed as a percent, not the grade.


Checking if a unit is already defined
-------------------------------------

The python ``in`` keyword works as expected with unit registries. Check if
a unit has been defined with the following:

.. doctest::

   >>> 'MHz' in ureg
   True
   >>> 'gigatrees' in ureg
   False
