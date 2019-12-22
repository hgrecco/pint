.. _tutorial:


Tutorial
========

Converting Quantities
---------------------

Pint has the concept of Unit Registry, an object within which units are defined
and handled. You start by creating your registry:

   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry()

.. testsetup:: *

   from pint import UnitRegistry
   ureg = UnitRegistry()
   Q_ = ureg.Quantity


If no parameter is given to the constructor, the unit registry is populated
with the default list of units and prefixes.
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

In this code `distance` and `time` are physical quantity objects (`Quantity`).
Physical quantities can be queried for their magnitude, units, and
dimensionality:

.. doctest::

   >>> print(distance.magnitude)
   24.0
   >>> print(distance.units)
   meter
   >>> print(distance.dimensionality)
   [length]

and can handle mathematical operations between:

.. doctest::

   >>> speed = distance / time
   >>> print(speed)
   3.0 meter / second

As unit registry knows about the relationship between different units, you can
convert quantities to the unit of choice:

.. doctest::

   >>> speed.to(ureg.inch / ureg.minute )
   <Quantity(7086.614173228345, 'inch / minute')>

This method returns a new object leaving the original intact as can be seen by:

.. doctest::

   >>> print(speed)
   3.0 meter / second

If you want to convert in-place (i.e. without creating another object), you can
use the `ito` method:

.. doctest::

   >>> speed.ito(ureg.inch / ureg.minute )
   >>> speed
   <Quantity(7086.614173228345, 'inch / minute')>
   >>> print(speed)
   7086.614173228345 inch / minute

If you ask Pint to perform an invalid conversion:

.. doctest::

   >>> speed.to(ureg.joule)
   Traceback (most recent call last):
   ...
   DimensionalityError: Cannot convert from 'inch / minute' ([length] / [time]) to 'joule' ([length] ** 2 * [mass] / [time] ** 2)

Sometimes, the magnitude of the quantity will be very large or very small.
The method 'to_compact' can adjust the units to make the quantity more
human-readable.

.. doctest::

   >>> wavelength = 1550 * ureg.nm
   >>> frequency = (ureg.speed_of_light / wavelength).to('Hz')
   >>> print(frequency)
   193414489032258.03 hertz
   >>> print(frequency.to_compact())
   193.41448903225802 terahertz

There are also methods 'to_base_units' and 'ito_base_units' which automatically
convert to the reference units with the correct dimensionality:

.. doctest::

   >>> height = 5.0 * ureg.foot + 9.0 * ureg.inch
   >>> print(height)
   5.75 foot
   >>> print(height.to_base_units())
   1.7526 meter
   >>> print(height)
   5.75 foot
   >>> height.ito_base_units()
   >>> print(height)
   1.7526 meter

There are also methods 'to_reduced_units' and 'ito_reduced_units' which perform
a simplified dimensional reduction, combining units with the same dimensionality
but otherwise keeping your unit definitions intact.

.. doctest::

   >>> density = 1.4 * ureg.gram / ureg.cm**3
   >>> volume = 10*ureg.cc
   >>> mass = density*volume
   >>> print(mass)
   14.0 cc * gram / centimeter ** 3
   >>> print(mass.to_reduced_units())
   14.0 gram
   >>> print(mass)
   14.0 cc * gram / centimeter ** 3
   >>> mass.ito_reduced_units()
   >>> print(mass)
   14.0 gram

If you want pint to automatically perform dimensional reduction when producing
new quantities, the UnitRegistry accepts a parameter `auto_reduce_dimensions`.
Dimensional reduction can be slow, so auto-reducing is disabled by default.

In some cases it is useful to define physical quantities objects using the
class constructor:

.. doctest::

   >>> Q_ = ureg.Quantity
   >>> Q_(1.78, ureg.meter) == 1.78 * ureg.meter
   True

(I tend to abbreviate Quantity as `Q_`) The built-in parser recognizes prefixed
and pluralized units even though they are not in the definition list:

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
   UndefinedUnitError: 'snail_speed' is not defined in the unit registry

You can add your own units to the registry or build your own list. More info on
that :ref:`defining`


String parsing
--------------

Pint can also handle units provided as strings:

.. doctest::

   >>> 2.54 * ureg.parse_expression('centimeter')
   <Quantity(2.54, 'centimeter')>

or using the registry as a callable for a short form for `parse_expression`:

.. doctest::

   >>> 2.54 * ureg('centimeter')
   <Quantity(2.54, 'centimeter')>

or using the `Quantity` constructor:

.. doctest::

   >>> Q_(2.54, 'centimeter')
   <Quantity(2.54, 'centimeter')>


Numbers are also parsed, so you can use an expression:

.. doctest::

   >>> ureg('2.54 * centimeter')
   <Quantity(2.54, 'centimeter')>

or:

.. doctest::

   >>> Q_('2.54 * centimeter')
   <Quantity(2.54, 'centimeter')>

or leave out the `*` altogether:

.. doctest::

   >>> Q_('2.54cm')
   <Quantity(2.54, 'centimeter')>

This enables you to build a simple unit converter in 3 lines:

.. doctest::

   >>> user_input = '2.54 * centimeter to inch'
   >>> src, dst = user_input.split(' to ')
   >>> Q_(src).to(dst)
   <Quantity(1.0, 'inch')>

Dimensionless quantities can also be parsed into an appropriate object:

.. doctest::

   >>> ureg('2.54')
   2.54
   >>> type(ureg('2.54'))
   <class 'float'>

or

.. doctest::

   >>> Q_('2.54')
   <Quantity(2.54, 'dimensionless')>
   >>> type(Q_('2.54'))
   <class 'pint.quantity.build_quantity_class.<locals>.Quantity'>

.. note:: Pint´s rule for parsing strings with a mixture of numbers and
   units is that **units are treated with the same precedence as numbers**.
   
For example, the unit of

.. doctest::

   >>> Q_('3 l / 100 km')
   <Quantity(0.03, 'kilometer * liter')>
   
may be unexpected first but is a consequence of applying this rule. Use
brackets to get the expected result:

.. doctest::

   >>> Q_('3 l / (100 km)')
   <Quantity(0.03, 'liter / kilometer')>

.. note:: Since version 0.7, Pint **does not** use eval_ under the hood.
   This change removes the `serious security problems`_ that the system is
   exposed to when parsing information from untrusted sources.


.. _sec-string-formatting:

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

Pint supports float formatting for numpy arrays as well:

.. doctest::

   >>> accel = np.array([-1.1, 1e-6, 1.2505, 1.3]) * ureg['meter/second**2']
   >>> # float formatting numpy arrays
   >>> print('The array is {:.2f}'.format(accel))
   The array is [-1.10 0.00 1.25 1.30] meter / second ** 2
   >>> # scientific form formatting with unit pretty printing
   >>> print('The array is {:+.2E~P}'.format(accel))
   The array is [-1.10E+00 +1.00E-06 +1.25E+00 +1.30E+00] m/s²
   
Pint also supports 'f-strings'_ from python>=3.6 :

.. doctest::

   >>> accel = 1.3 * ureg['meter/second**2'] 
   >>> print(f'The str is {accel}')
   The str is 1.3 meter / second ** 2
   >>> print(f'The str is {accel:.3e}')
   The str is 1.300e+00 meter / second ** 2
   >>> print(f'The str is {accel:~}')
   The str is 1.3 m / s ** 2
   >>> print(f'The str is {accel:~.3e}')
   The str is 1.300e+00 m / s ** 2
   >>> print(f'The str is {accel:~H}')
   The str is 1.3 m/s²  

But Pint also extends the standard formatting capabilities for unicode and
LaTeX representations:

.. doctest::

   >>> accel = 1.3 * ureg['meter/second**2']
   >>> # Pretty print
   >>> 'The pretty representation is {:P}'.format(accel)
   'The pretty representation is 1.3 meter/second²'
   >>> # Latex print
   >>> 'The latex representation is {:L}'.format(accel)
   'The latex representation is 1.3\\ \\frac{\\mathrm{meter}}{\\mathrm{second}^{2}}'
   >>> # HTML print
   >>> 'The HTML representation is {:H}'.format(accel)
   'The HTML representation is 1.3 meter/second<sup>2</sup>'

If you want to use abbreviated unit names, prefix the specification with `~`:

.. doctest::

   >>> 'The str is {:~}'.format(accel)
   'The str is 1.3 m / s ** 2'
   >>> 'The pretty representation is {:~P}'.format(accel)
   'The pretty representation is 1.3 m/s²'


The same is true for latex (`L`) and HTML (`H`) specs.

.. note::
   The abbreviated unit is drawn from the unit registry where the 3rd item in the 
   equivalence chain (ie 1 = 2 = **3**) will be returned when the prefix '~' is 
   used. The 1st item in the chain is the canonical name of the unit.

The formatting specs (ie 'L', 'H', 'P') can be used with Python string 'formatting 
syntax'_ for custom float representations. For example, scientific notation:

.. doctest::
   >>> 'Scientific notation: {:.3e~L}'.format(accel)
   'Scientific notation: 1.300\\times 10^{0}\\ \\frac{\\mathrm{m}}{\\mathrm{s}^{2}}'

Pint also supports the LaTeX siunitx package:

.. doctest::

   >>> accel = 1.3 * ureg['meter/second**2']
   >>> # siunitx Latex print
   >>> print('The siunitx representation is {:Lx}'.format(accel))
   The siunitx representation is \SI{1.3}{\meter\per\second\squared}
   >>> accel = accel.plus_minus(0.2)
   >>> print('The siunitx representation is {:Lx}'.format(accel))
   The siunitx representation is \SI{1.3 +- 0.2}{\meter\per\second\squared}

Additionally, you can specify a default format specification:

.. doctest::

   >>> 'The acceleration is {}'.format(accel)
   'The acceleration is 1.3 meter / second ** 2'
   >>> ureg.default_format = 'P'
   >>> 'The acceleration is {}'.format(accel)
   'The acceleration is 1.3 meter/second²'


Finally, if Babel_ is installed you can translate unit names to any language

.. doctest::

   >>> accel.format_babel(locale='fr_FR')
   '1.3 mètre par seconde²'


Using Pint in your projects
---------------------------

If you use Pint in multiple modules within your Python package, you normally
want to avoid creating multiple instances of the unit registry.
The best way to do this is by instantiating the registry in a single place. For
example, you can add the following code to your package `__init__.py`::

   from pint import UnitRegistry
   ureg = UnitRegistry()
   Q_ = ureg.Quantity


Then in `yourmodule.py` the code would be::

   from . import ureg, Q_

   length = 10 * ureg.meter
   my_speed = Q_(20, 'm/s')

If you are pickling and unplicking Quantities within your project, you should
also define the registry as the application registry::

   from pint import UnitRegistry, set_application_registry
   ureg = UnitRegistry()
   set_application_registry(ureg)


.. warning:: There are no global units in Pint. All units belong to a registry and you can have multiple registries instantiated at the same time. However, you are not supposed to operate between quantities that belong to different registries. Never do things like this:

.. doctest::

   >>> q1 = 10 * UnitRegistry().meter
   >>> q2 = 10 * UnitRegistry().meter
   >>> q1 + q2
   Traceback (most recent call last):
   ...
   ValueError: Cannot operate with Quantity and Quantity of different registries.
   >>> id(q1._REGISTRY) == id(q2._REGISTRY)
   False


.. _eval: http://docs.python.org/3/library/functions.html#eval
.. _`serious security problems`: http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
.. _`Babel`: http://babel.pocoo.org/
.. _'formatting syntax': https://docs.python.org/3/library/string.html#format-specification-mini-language
.. _'f-strings': https://www.python.org/dev/peps/pep-0498/