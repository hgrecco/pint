.. _defining:

Defining units
==============

Programmatically
----------------

You can easily add units to the registry programmatically. Let's add a dog_year (sometimes writen as dy) equivalent to 52 (human) days::

   >>> from pint import UnitRegistry
   # We first instantiate the registry.
   # If we do not provide any parameter, the default unit definitions are used.
   >>> ureg = UnitRegistry()
   >>> Q_ = ureg.Quantity

   # Here we add the unit
   >>> ureg.add_unit('dog_year', Q_(52, 'day'), ('dy', ))

   # We create a quantity based on that unit and we convert to years.
   >>> lassie_lifespan = Q_(10, 'dog_years')
   >>> print(lassie_lifespan.to('year'))

Note that we have used the name `dog_years` even though we have not defined the plural form as an alias. Pint takes care of that, so you don't have to.

Units added programmatically are forgotten when the UnitRegistry object is deleted.


In a definition file
--------------------

To define units in a persistent way you need to create a unit definition file. Such files are simple text files in which the units are defined as function of other units. For example this is how the minute and the hour are defined in `default_en.txt`::

    hour = 60 * minute = h = hr
    minute = 60 * second = min

It is quite straightforward, isn't it? We are saying that `minute` is `60 seconds` and is also known as `min`. The first word is always the canonical name. Next comes the definition (based on other units). Finally, a list of aliases, separated by equal signs.

The order in which units are defined does not matter, Pint will resolve the dependencies to define them in the right order. What is important is that if you transverse all definitions, a reference unit is reached. A reference unit is not defined as a function of another units but of a dimension. For the time in `default_en.txt`, this is the `second`::

    second = [time] = s = sec

By defining `second` as equal to a string `time` in square brackets we indicate that:

 * `time` is a physical dimension.
 * `second` is a reference unit.

The ability to define basic physical dimensions as well as reference units allows to construct arbitrary units systems.

Pint is shipped with a default definition file named `default_en.txt` where `en` stands for english. You can add your own definitions to the end of this file but you will have to be careful to merge when you update Pint. An easier way is to create a new file (e.g. `mydef.txt`) with your definitions::

   dog_year = 52 * day = dy

and then in Python, you can load it as::

   >>> from pint import UnitRegistry
   # First we create the registry.
   >>> ureg = UnitRegistry()
   # Then we append the new definitions
   >>> ureg.define_from_file('/your/path/to/my_def.txt')

If you make a translation of the default units, you don't want to append the translated definitions so you just give the filename to the constructor::

   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry('/your/path/to/default_es.txt')


Prefixes
--------

You can also add prefixes programmatically::

   >>> ureg.add_prefix('myprefix', 30, 'my')

where the number indicates the multiplication factor.

In the definition file, prefixes are identified by a trailing dash::

   yocto- = 10.0**-24 = y-

It is important to note that prefixed defined in this way can be used with any unit, including non-metric ones (e.g. kiloinch is valid for Pint). This simplifies definitions files enormously without introducing major problems. Pint, like Python, believes that we are all consenting adults.
