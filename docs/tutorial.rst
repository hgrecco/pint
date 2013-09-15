.. _tutorial:


Tutorial
========

Converting Quantities
---------------------

Pint has the concept of Unit Registry, an object within which units are defined and handled. You start by creating your registry::

   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry()

.. testsetup:: *

   from pint import UnitRegistry
   ureg = UnitRegistry()
   Q_ = ureg.Quantity

If no parameter is given to the constructor, the unit registry is populated with the default list of units and prefixes.
You can now simply use the registry in the following way:

.. doctest::

   >>> distance = 24.0 * ureg.meter
   >>> print(distance)
   24.0 meter
   >>> time = 8.0 * ureg.second
   >>> print(time)
   8.0 second
   >>> print(repr(time))
   <Quantity(8.0, 'second')>

In this code `distance` and `time` are physical quantities objects (`Quantity`). Physical quantities can be queried for the magnitude and units:

.. doctest::

   >>> print(distance.magnitude)
   24.0
   >>> print(distance.units)
   meter

and can handle mathematical operations between:

.. doctest::

   >>> speed = distance / time
   >>> print(speed)
   3.0 meter / second

As unit registry knows about the relationship between different units, you can convert quantities to the unit of choice:

.. doctest::

   >>> speed.to(ureg.inch / ureg.minute )
   <Quantity(7086.614173228345, 'inch / minute')>

This method returns a new object leaving the original intact as can be seen by:

.. doctest::

   >>> print(speed)
   3.0 meter / second

If you want to convert in-place (i.e. without creating another object), you can use the `ito` method:

.. doctest::

   >>> speed.ito(ureg.inch / ureg.minute )
   <Quantity(7086.614173228345, 'inch / minute')>
   >>> print(speed)
   7086.614173228345 inch / minute

If you ask Pint to perform and invalid conversion:

.. doctest::

   >>> speed.to(ureg.joule)
   Traceback (most recent call last):
   ...
   pint.pint.DimensionalityError: Cannot convert from 'inch / minute' (length / time) to 'joule' (length ** 2 * mass / time ** 2)


In some cases it is useful to define physical quantities objects using the class constructor:

.. doctest::

   >>> Q_ = ureg.Quantity
   >>> Q_(1.78, ureg.meter) == 1.78 * ureg.meter
   True

(I tend to abbreviate Quantity as `Q_`) The in-built parse allows to recognize prefixed and pluralized units even though they are not in the definition list:

.. doctest::

   >>> distance = 42 * ureg.kilometers
   >>> print(distance)
   42 kilometer
   >>> print(distance.to(ureg.meter))
   42000.0 meter

If you try to use a unit which is not in the registry:

.. doctest::

   >>> speed = 23 * ureg.snail_speed
   Traceback (most recent call last):
   ...
   pint.pint.UndefinedUnitError: 'snail_speed' is not defined in the unit registry

You can add your own units to the registry or build your own list. More info on that :ref:`defining`


String parsing
--------------

Pint can also handle units provided as strings:

.. doctest::

   >>> 2.54 * ureg['centimeter']
   <Quantity(2.54, 'centimeter')>

or via de `Quantity` constructor:

.. doctest::

   >>> Q_(2.54, 'centimeter')
   <Quantity(2.54, 'centimeter')>

Numbers are also parsed:

.. doctest::

   >>> Q_('2.54 * centimeter')
   <Quantity(2.54, 'centimeter')>

This enables you to build a simple unit converter in 3 lines:

.. doctest:

   >>> user_input = '2.54 * centimeter to inch'
   >>> src, dst = user_input.split(' to ')
   >>> Q_(src).to(dst)
   <Quantity(1.0, 'inch')>

Take a look at `qconvert.py` within the examples folder for a full script.


String formatting
-----------------

Pint's physical quantities can be easily printed:

.. doctest::

   >>> accel = 1.3 * ureg['meter/second**2']
   >>> # The standard string formatting code
   >>> print('The str is {!s}'.format(accel))
   The str is 1.3 meter / second ** 2
   >>> # The standard representation formatting code
   >>> print('The repr is {!r}'.format(accel))
   The repr is <Quantity(1.3, 'meter / second ** 2')>
   >>> # Accessing useful attributes
   >>> print('The magnitude is {0.magnitude} with units {0.units}'.format(accel))
   The magnitude is 1.3 with units meter / second ** 2

But Pint also extends the standard formatting capabilities for unicode and latex representations:

.. doctest::

   >>> accel = 1.3 * ureg['meter/second**2']
   >>> # Pretty print
   >>> 'The pretty representation is {:P}'.format(accel)
   'The pretty representation is 1.3 meter/second²'
   >>> # Latex print
   >>> 'The latex representation is {:L}'.format(accel)
   'The latex representation is 1.3 \\frac{meter}{second^{2}}'
   >>> # HTML print
   >>> 'The latex representation is {:H}'.format(accel)
   'The latex representation is 1.3 meter/second<sup>2</sup>'

If you want to use abbreviated unit names, suffix the specification with `~`:

.. doctest::

   >>> 'The str is {:~}'.format(accel)
   'The str is 1.3 m / s ** 2'

The same is true for latex (`L`), pretty (`P`) and HTML (`H`) specs.


Using Pint in your projects
---------------------------

If you use Pint in multiple modules within you Python package, you normally want to avoid creating multiple instances of the unit registry.
The best way to do this is by instantiating the registry in a single place. For example, you can add the following code to your package `__init__.py`::

   from pint import UnitRegistry
   ureg = UnitRegistry()
   Q_ = ureg.Quantity


Then in `yourmodule.py` the code would be::

   from . import ureg, Q_

   length = 10 * ureg.meter
   my_speed = Quantity(20, 'm/s')


.. warning:: There are no global units in Pint. All units belong to a registry and you can have multiple registries instantiated at the same time. However, you are not supposed to operate between quantities that belong to different registries. Never do things like this::

    >>> q1 = UnitRegistry().meter
    >>> q2 = UnitRegistry().meter
    >>> # q1 and q2 belong to different registries!
    >>> id(q1._REGISTRY) is id(q2._REGISTRY) # False

