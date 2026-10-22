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


In a TOML definition file
-------------------------

Definitions can also be written in TOML. Pint ships ``default_en.toml``, which
holds the same definitions as ``default_en.txt``. A file is read as TOML when
its name ends in ``.toml``::

    [defaults]
    group = "international"
    system = "mks"

    [prefix]
    kilo = { value = "1e3", defined_symbol = "k" }

    [dimension]
    density = { value = "[mass] / [volume]" }

    [unit]
    meter = { value = "[length]", defined_symbol = "m", aliases = ["metre"] }
    minute = { value = "60 * second", defined_symbol = "min" }
    degree_Celsius = { value = "kelvin; offset: 273.15", defined_symbol = "°C" }

    [[group]]
    name = "Avoirdupois"
    [group.definitions]
    pound = { value = "7000 * grain", defined_symbol = "lb" }

    [[system]]
    name = "mks"
    using_group_names = ["international"]
    rules = ["meter", "kilogram", "second"]

    [[context]]
    name = "spectroscopy"
    aliases = ["sp"]
    defaults = { n = 1 }
    relations = ["[length] <-> [frequency]: speed_of_light / n / value"]

Values are written as strings, using the same syntax as the text format. Every
section is optional. There is no ``@alias`` or ``@import``: list the aliases on
the unit, and put all definitions in one file.

A registry's definitions can be saved to a TOML file (this needs ``tomlkit``)::

   >>> ureg.write_definitions('/your/path/to/my_def.toml') # doctest: +SKIP

Groups, systems and contexts made with Python objects rather than definitions
cannot be saved, and Pint warns about them.


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
