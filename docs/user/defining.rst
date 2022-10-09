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


Checking if a unit is already defined
-------------------------------------

The python ``in`` keyword works as expected with unit registries. Check if
a unit has been defined with the following:

.. doctest::

   >>> 'MHz' in ureg
   True
   >>> 'gigatrees' in ureg
   False
