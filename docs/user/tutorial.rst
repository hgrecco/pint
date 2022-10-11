
Tutorial
========

Follow the steps below and learn how to use Pint to track physical quantities
and perform unit conversions in Python.

Initializing a Registry
-----------------------

Before using Pint, initialize a :class:`UnitRegistry() <pint.registry.UnitRegistry>`
object. The ``UnitRegistry`` stores the unit definitions, their relationships,
and handles conversions between units.

.. doctest::

   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry()

If no parameters are given to the constructor, the ``UnitRegistry`` is populated
with the `default list of units`_ and prefixes.

Defining a Quantity
-------------------

Once you've initialized your ``UnitRegistry``, you can define quantities easily:

.. doctest::

   >>> distance = 24.0 * ureg.meter
   >>> distance
   <Quantity(24.0, 'meter')>
   >>> print(distance)
   24.0 meter

As you can see, ``distance`` here is a :class:`Quantity() <pint.quantity.Quantity>`
object that represents a physical quantity. Quantities can be queried for their
magnitude, units, and dimensionality:

.. doctest::

   >>> distance.magnitude
   24.0
   >>> distance.units
   <Unit('meter')>
   >>> print(distance.dimensionality)
   [length]

and can correctly handle many mathematical operations, including with other
:class:`Quantity() <pint.quantity.Quantity>` objects:

.. doctest::

   >>> time = 8.0 * ureg.second
   >>> print(time)
   8.0 second
   >>> speed = distance / time
   >>> speed
   <Quantity(3.0, 'meter / second')>
   >>> print(speed)
   3.0 meter / second
   >>> print(speed.dimensionality)
   [length] / [time]

Notice the built-in parser recognizes prefixed and pluralized units even though
they are not in the definition list:

.. doctest::

   >>> distance = 42 * ureg.kilometers
   >>> print(distance)
   42 kilometer
   >>> print(distance.to(ureg.meter))
   42000.0 meter

Pint will complain if you try to use a unit which is not in the registry:

.. doctest::

   >>> speed = 23 * ureg.snail_speed
   Traceback (most recent call last):
   ...
   UndefinedUnitError: 'snail_speed' is not defined in the unit registry

You can add your own units to the existing registry, or build your own list.
See the page on :ref:`defining` for more information on that.

See `String parsing`_ and :doc:`defining-quantities` for more ways of defining
a ``Quantity()`` object.

``Quantity()`` objects also work well with NumPy arrays, which you can
read about in the section on :doc:`NumPy support <numpy>`.

Converting to different units
-----------------------------

As the underlying ``UnitRegistry`` knows the relationships between
different units, you can convert a ``Quantity`` to the units of your choice using
the ``to()`` method, which accepts a string or a :class:`Unit() <pint.unit.Unit>` object:

.. doctest::

   >>> speed.to('inch/minute')
   <Quantity(7086.61417, 'inch / minute')>
   >>> ureg.inch / ureg.minute
   <Unit('inch / minute')>
   >>> speed.to(ureg.inch / ureg.minute)
   <Quantity(7086.61417, 'inch / minute')>

This method returns a new object leaving the original intact as can be seen by:

.. doctest::

   >>> print(speed)
   3.0 meter / second

If you want to convert in-place (i.e. without creating another object), you can
use the ``ito()`` method:

.. doctest::

   >>> speed.ito(ureg.inch / ureg.minute)
   >>> speed
   <Quantity(7086.61417, 'inch / minute')>
   >>> print(speed)
   7086.6141... inch / minute

Pint will complain if you ask it to perform a conversion it doesn't know
how to do:

.. doctest::

   >>> speed.to(ureg.joule)
   Traceback (most recent call last):
   ...
   DimensionalityError: Cannot convert from 'inch / minute' ([length] / [time]) to 'joule' ([length] ** 2 * [mass] / [time] ** 2)

See the section on :doc:`contexts` for information about expanding Pint's
automatic conversion capabilities for your application.

Simplifying units
-----------------

Sometimes, the magnitude of the quantity will be very large or very small.
The method ``to_compact()`` can adjust the units to make a quantity more
human-readable:

.. doctest::

   >>> wavelength = 1550 * ureg.nm
   >>> frequency = (ureg.speed_of_light / wavelength).to('Hz')
   >>> print(frequency)
   193414489032258.03 hertz
   >>> print(frequency.to_compact())
   193.414489032... terahertz

There are also methods ``to_base_units()`` and ``ito_base_units()`` which automatically
convert to the reference units with the correct dimensionality:

.. doctest::

   >>> height = 5.0 * ureg.foot + 9.0 * ureg.inch
   >>> print(height)
   5.75 foot
   >>> print(height.to_base_units())
   1.752... meter
   >>> print(height)
   5.75 foot
   >>> height.ito_base_units()
   >>> print(height)
   1.752... meter

There are also methods ``to_reduced_units()`` and ``ito_reduced_units()`` which perform
a simplified dimensional reduction, combining units with the same dimensionality
but otherwise keeping your unit definitions intact.

.. doctest::

   >>> density = 1.4 * ureg.gram / ureg.cm**3
   >>> volume = 10*ureg.cc
   >>> mass = density*volume
   >>> print(mass)
   14.0 cubic_centimeter * gram / centimeter ** 3
   >>> print(mass.to_reduced_units())
   14.0 gram
   >>> print(mass)
   14.0 cubic_centimeter * gram / centimeter ** 3
   >>> mass.ito_reduced_units()
   >>> print(mass)
   14.0 gram

If you want pint to automatically perform dimensional reduction when producing
new quantities, the ``UnitRegistry`` class accepts a parameter ``auto_reduce_dimensions``.
Dimensional reduction can be slow, so auto-reducing is disabled by default.

String parsing
--------------

Pint includes powerful string parsing for identifying magnitudes and units. In
many cases, units can be defined as strings:

.. doctest::

   >>> 2.54 * ureg('centimeter')
   <Quantity(2.54, 'centimeter')>

or using the ``Quantity`` constructor:

.. doctest::

   >>> Q_ = ureg.Quantity
   >>> Q_(2.54, 'centimeter')
   <Quantity(2.54, 'centimeter')>

Numbers are also parsed, so you can use an expression:

.. doctest::

   >>> ureg('2.54 * centimeter')
   <Quantity(2.54, 'centimeter')>
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

Strings containing values can be parsed using the ``ureg.parse_pattern()`` function.
A ``format``-like string with the units defined in it is used as the pattern:

.. doctest::

   >>> input_string = '10 feet 10 inches'
   >>> pattern = '{feet} feet {inch} inches'
   >>> ureg.parse_pattern(input_string, pattern)
   [<Quantity(10.0, 'foot')>, <Quantity(10.0, 'inch')>]

To search for multiple matches, set the ``many`` parameter to ``True``. The following
example also demonstrates how the parser is able to find matches in amongst filler characters:

.. doctest::

   >>> input_string = '10 feet - 20 feet ! 30 feet.'
   >>> pattern = '{feet} feet'
   >>> ureg.parse_pattern(input_string, pattern, many=True)
   [[<Quantity(10.0, 'foot')>], [<Quantity(20.0, 'foot')>], [<Quantity(30.0, 'foot')>]]

The full power of regex can also be employed when writing patterns:

.. doctest::

   >>> input_string = "10` - 20 feet ! 30 ft."
   >>> pattern = r"{feet}(`| feet| ft)"
   >>> ureg.parse_pattern(input_string, pattern, many=True)
   [[<Quantity(10.0, 'foot')>], [<Quantity(20.0, 'foot')>], [<Quantity(30.0, 'foot')>]]

*Note that the curly brackets (``{}``) are converted to a float-matching pattern by the parser.*

This function is useful for tasks such as bulk extraction of units from thousands
of uniform strings or even very large texts with units dotted around in no particular pattern.


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

   >>> import numpy as np
   >>> accel = np.array([-1.1, 1e-6, 1.2505, 1.3]) * ureg['meter/second**2']
   >>> # float formatting numpy arrays
   >>> print('The array is {:.2f}'.format(accel))
   The array is [-1.10 0.00 1.25 1.30] meter / second ** 2
   >>> # scientific form formatting with unit pretty printing
   >>> print('The array is {:+.2E~P}'.format(accel))
   The array is [-1.10E+00 +1.00E-06 +1.25E+00 +1.30E+00] m/s²

Pint also supports `f-strings`_ from python>=3.6 :

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
   >>> print(f'The str is {accel:~H}')      # HTML format (displays well in Jupyter)
   The str is 1.3 m/s<sup>2</sup>

But Pint also extends the standard formatting capabilities for unicode, LaTeX, and HTML
representations:

.. doctest::

   >>> accel = 1.3 * ureg['meter/second**2']
   >>> # Pretty print
   >>> 'The pretty representation is {:P}'.format(accel)
   'The pretty representation is 1.3 meter/second²'
   >>> # LaTeX print
   >>> 'The LaTeX representation is {:L}'.format(accel)
   'The LaTeX representation is 1.3\\ \\frac{\\mathrm{meter}}{\\mathrm{second}^{2}}'
   >>> # HTML print - good for Jupyter notebooks
   >>> 'The HTML representation is {:H}'.format(accel)
   'The HTML representation is 1.3 meter/second<sup>2</sup>'

If you want to use abbreviated unit names, prefix the specification with `~`:

.. doctest::

   >>> 'The str is {:~}'.format(accel)
   'The str is 1.3 m / s ** 2'
   >>> 'The pretty representation is {:~P}'.format(accel)
   'The pretty representation is 1.3 m/s²'


The same is true for LaTeX (`L`) and HTML (`H`) specs.

.. note::
   The abbreviated unit is drawn from the unit registry where the 3rd item in the
   equivalence chain (ie 1 = 2 = **3**) will be returned when the prefix '~' is
   used. The 1st item in the chain is the canonical name of the unit.

The formatting specs (ie 'L', 'H', 'P') can be used with Python string
`formatting syntax`_ for custom float representations. For example, scientific
notation:

.. doctest::

   >>> 'Scientific notation: {:.3e~L}'.format(accel)
   'Scientific notation: 1.300\\times 10^{0}\\ \\frac{\\mathrm{m}}{\\mathrm{s}^{2}}'

Pint also supports the LaTeX `siunitx` package:

.. doctest::
   :skipif: not_installed['uncertainties']

   >>> accel = 1.3 * ureg['meter/second**2']
   >>> # siunitx Latex print
   >>> print('The siunitx representation is {:Lx}'.format(accel))
   The siunitx representation is \SI[]{1.3}{\meter\per\second\squared}
   >>> accel = accel.plus_minus(0.2)
   >>> print('The siunitx representation is {:Lx}'.format(accel))
   The siunitx representation is \SI{1.30 +- 0.20}{\meter\per\second\squared}

Additionally, you can specify a default format specification:

.. doctest::

   >>> accel = 1.3 * ureg['meter/second**2']
   >>> 'The acceleration is {}'.format(accel)
   'The acceleration is 1.3 meter / second ** 2'
   >>> ureg.default_format = 'P'
   >>> 'The acceleration is {}'.format(accel)
   'The acceleration is 1.3 meter/second²'


Localizing
----------

If Babel_ is installed you can translate unit names to any language

.. doctest::

   >>> accel.format_babel(locale='fr_FR')
   '1.3 mètre par seconde²'

You can also specify the format locale at the registry level either at creation:

.. doctest::

    >>> ureg = UnitRegistry(fmt_locale='fr_FR')

or later:

.. doctest::

    >>> ureg.set_fmt_locale('fr_FR')

and by doing that, string formatting is now localized:

.. doctest::

    >>> accel = 1.3 * ureg['meter/second**2']
    >>> str(accel)
    '1.3 mètre par seconde²'
    >>> "%s" % accel
    '1.3 mètre par seconde²'
    >>> "{}".format(accel)
    '1.3 mètre par seconde²'


Using Pint in your projects
---------------------------

If you use Pint in multiple modules within your Python package, you normally
want to avoid creating multiple instances of the unit registry.
The best way to do this is by instantiating the registry in a single place. For
example, you can add the following code to your package ``__init__.py``

.. code-block:: python

   from pint import UnitRegistry
   ureg = UnitRegistry()
   Q_ = ureg.Quantity


Then in ``yourmodule.py`` the code would be

.. code-block:: python

   from . import ureg, Q_

   length = 10 * ureg.meter
   my_speed = Q_(20, 'm/s')

If you are pickling and unpickling Quantities within your project, you should
also define the registry as the application registry

.. code-block:: python

   from pint import UnitRegistry, set_application_registry
   ureg = UnitRegistry()
   set_application_registry(ureg)


.. warning:: There are no global units in Pint. All units belong to a registry and
    you can have multiple registries instantiated at the same time. However, you
    are not supposed to operate between quantities that belong to different registries.
    Never do things like this:

.. doctest::

   >>> q1 = 10 * UnitRegistry().meter
   >>> q2 = 10 * UnitRegistry().meter
   >>> q1 + q2
   Traceback (most recent call last):
   ...
   ValueError: Cannot operate with Quantity and Quantity of different registries.
   >>> id(q1._REGISTRY) == id(q2._REGISTRY)
   False


.. _`default list of units`: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
.. _`Babel`: http://babel.pocoo.org/
.. _`formatting syntax`: https://docs.python.org/3/library/string.html#format-specification-mini-language
.. _`f-strings`: https://www.python.org/dev/peps/pep-0498/
