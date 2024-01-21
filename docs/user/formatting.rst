.. currentmodule:: pint


.. ipython:: python
   :suppress:

   import pint


String formatting specification
===============================

The conversion of :py:class:`Unit`, :py:class:`Quantity` and :py:class:`Measurement`
objects to strings (e.g. through the :py:class:`str` builtin or f-strings) can be
customized using :ref:`format specifications <formatspec>`. The basic format is:

.. code-block:: none

   [magnitude format][modifier][pint format]

where each part is optional and the order of these is arbitrary.

.. ipython::

    q = 1e-6 * u

    # modifiers
    f"{q:~P}"  # short pretty
    f"{q:~#P}"  # compact short pretty
    f"{q:P#~}"  # also compact short pretty

    # additional magnitude format
    f"{q:.2f~#P}"  # short compact pretty with 2 float digits
    f"{q:#~}"  # short compact default


In case the format is omitted, the corresponding value in the formatter
``.default_format`` attribute is filled in. For example:

   ureg.formatter.default_format = "P"
   f"{q}"


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

.. ipython::

   In [1]: u = ureg.Unit("m ** 3 / (s ** 2 * kg)")

   In [2]: @pint.register_unit_format("simple")
      ...: def format_unit_simple(unit, registry, **options):
      ...:     return " * ".join(f"{u} ** {p}" for u, p in unit.items())

   In [3]: f"{u:~simple}"

where ``unit`` is a :py:class:`dict` subclass containing the unit names and
their exponents.

You can choose to replace the complete formatter. Briefly, the formatter if an object with the
following methods: `format_magnitude`, `format_unit`, `format_quantity`, `format_uncertainty`,
`format_measurement`. The easiest way to create your own formatter is to subclass one that you like.

.. ipython::

   In [1]: from pint.delegates.formatter.plain import DefaultFormatter, PlainUnit

   In [2]: class MyFormatter(DefaultFormatter):
      ...:
      ...:     def format_unit(self, unit: PlainUnit, uspec: str = "", **babel_kwds) -> str:
      ...:         return "ups!"

   In [3]: ureg.formatter = MyFormatter()

   In [4]: str(1e-6 * u)
