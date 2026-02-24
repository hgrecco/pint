.. currentmodule:: pint


String formatting specification
===============================

The conversion of :py:class:`Unit`, :py:class:`Quantity` and :py:class:`Measurement`
objects to strings (e.g. through the :py:class:`str` builtin or f-strings) can be
customized using :ref:`format specifications <formatspec>`. The basic format is:

.. code-block:: none

   [magnitude format][modifier][pint format]

where each part is optional and the order of these is arbitrary.

.. doctest::

   >>> import pint
   >>> ureg = pint.UnitRegistry()
   >>> q = 2.3e-6 * ureg.m ** 3 / (ureg.s ** 2 * ureg.kg)
   >>> f"{q:~P}"  # short pretty
   '2.3×10⁻⁶ m³/kg/s²'
   >>> f"{q:~^P}"  # short pretty with negative exponents
   '2.3×10⁻⁶ kg⁻¹·m³·s⁻²'
   >>> f"{q:~#P}"  # compact short pretty
   '2.3 mm³/g/s²'
   >>> f"{q:P#~}"  # also compact short pretty
   '2.3 mm³/g/s²'
   >>> f"{q:.2f~#P}"  # short compact pretty with 2 float digits
   '2.30 mm³/g/s²'
   >>> f"{q:#~}"  # short compact default
   '2.3 mm ** 3 / g / s ** 2'

In case the format is omitted, the corresponding value in the formatter
``.default_format`` attribute is filled in. For example:

.. doctest::

   >>> ureg.formatter.default_format = "P"
   >>> f"{q}"
   '2.3×10⁻⁶ meter³/kilogram/second²'

Pint Format Types
-----------------
``pint`` comes with a variety of unit formats. These impact the complete representation:

======= =============== ======================================================================
Spec    Name            Examples
======= =============== ======================================================================
``D``   default         ``3.4e+09 kilogram * meter / second ** 2``
``P``   pretty          ``3.4×10⁹ kilogram·meter/second²``
``H``   HTML            ``3.4×10<sup>9</sup> kilogram meter/second<sup>2</sup>``
``L``   latex           ``3.4\\times 10^{9}\\ \\frac{\\mathrm{kilogram} \\cdot \\mathrm{meter}}{\\mathrm{second}^{2}}``
``Lx``  latex siunitx   ``\\SI[]{3.4e+09}{\\kilo\\gram\\meter\\per\\second\\squared}``
``C``   compact         ``3.4e+09 kilogram*meter/second**2``
======= =============== ======================================================================

These examples are using `g`` as numeric modifier. :py:class:`Measurement` are also affected
by these modifiers.


Quantity modifiers
------------------

======== =================================================== ================================
Modifier Meaning                                             Example
======== =================================================== ================================
``#``    Call :py:meth:`Quantity.to_compact` first           ``1.0 m·mg/s²`` (``f"{q:#~P}"``)
======== =================================================== ================================

Unit modifiers
--------------

======== =================================================== ================================
Modifier Meaning                                             Example
======== =================================================== ================================
``~``    Use the unit's symbol instead of its canonical name ``kg·m/s²`` (``f"{u:~P}"``)
``^``    Use negative exponents instead of ratio             ``kg·m·s⁻²`` (``f"{u:~^P}"``)
======== =================================================== ================================

Magnitude modifiers
-------------------

Pint uses the :ref:`format specifications <formatspec>`. However, it is important to remember
that only the  type honors the locale. Using any other numeric format (e.g. `g`, `e`, `f`)
will result  in a non-localized representation of the number.


Custom formats
--------------
Using :py:func:`pint.register_unit_format`, it is possible to add custom
formats:

.. doctest::

   >>> @pint.register_unit_format("Z")
   ... def format_unit_simple(unit, registry, **options):
   ...     return " * ".join(f"{u} ** {p}" for u, p in unit.items())
   >>> f"{q:Z}"
   '2.3e-06 kilogram ** -1 * meter ** 3 * second ** -2'

where ``unit`` is a :py:class:`dict` subclass containing the unit names and
their exponents, ``registry`` is the current instance of :py:class:``UnitRegistry`` and
``options`` is not yet implemented.

You can choose to replace the complete formatter. Briefly, the formatter if an object with the
following methods: `format_magnitude`, `format_unit`, `format_quantity`, `format_uncertainty`,
`format_measurement`. The easiest way to create your own formatter is to subclass one that you
like and replace the methods you need. For example, to replace the unit formatting:

.. doctest::

   >>> from pint.delegates.formatter.plain import DefaultFormatter
   >>> class MyFormatter(DefaultFormatter):
   ...
   ...      default_format = ""
   ...
   ...      def format_unit(self, unit, uspec, sort_func, **babel_kwds) -> str:
   ...          return "ups!"
   ...
   >>> ureg.formatter = MyFormatter()
   >>> ureg.formatter._registry = ureg
   >>> str(q)
   '2.3e-06 ups!'


By replacing other methods, you can customize the output as much as you need.

SciForm_ is a library that can be used to format the magnitude of the number. This can be used
in a customer formatter as follows:

.. doctest::

   >>> from sciform import Formatter
   >>> sciform_formatter = Formatter(round_mode="sig_fig", ndigits=4, exp_mode="engineering")

   >>> class MyFormatter(DefaultFormatter):
   ...
   ...      default_format = ""
   ...
   ...      def format_magnitude(self, value, spec, **options) -> str:
   ...          return sciform_formatter(value)
   ...
   >>> ureg.formatter = MyFormatter()
   >>> ureg.formatter._registry = ureg
   >>> str(q * 10)
   '23.00e-06 meter ** 3 / second ** 2 / kilogram'


.. _SciForm: https://sciform.readthedocs.io/en/stable/
