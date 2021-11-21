.. _formatting:
.. currentmodule:: pint


.. ipython:: python
   :suppress:

   import pint


String formatting
=================
The conversion of :py:class:`Unit` and :py:class:`Quantity` objects to strings (e.g.
through the :py:class:`str` builtin or f-strings) can be customized using :ref:`format
specifications <formatspec>`. The basic format is:

.. code-block:: none

   [magnitude format][modifier][unit format]

where each part is optional and the order of these is arbitrary.

In case any part (except the modifier) is omitted, the corresponding value in
:py:attr:`Quantity.default_format` or :py:attr:`Unit.default_format` is filled in. If
that is not set (it evaluates to ``False``), :py:attr:`UnitRegistry.default_format` is
used.  If both are not set, the global default of ``"D"`` and the magnitude's default
format are used instead.

.. note::

   Modifiers may be used without specifying any format: ``"~"`` is a valid format
   specification.


Unit Format Specifications
--------------------------
The :py:class:`Unit` class ignores the magnitude format part, and the unit format
consists of just the format type.

Let's look at some examples:

.. ipython:: python

   ureg = pint.UnitRegistry()
   u = ureg.kg * ureg.m / ureg.s ** 2

   f"{u:P}"  # using the pretty format
   f"{u:~P}"  # short pretty
   f"{u:P~}"  # also short pretty

   # default format
   u.default_format
   ureg.default_format
   str(u)  # default: default
   f"{u:~}"  # default: short default
   ureg.default_format = "C"  # registry default to compact
   str(u)  # default: compact
   f"{u}"  # default: compact
   u.default_format = "P"
   f"{u}"  # default: pretty
   u.default_format = ""  # TODO: switch to None
   ureg.default_format = ""  # TODO: switch to None
   f"{u}"  # default: default

Unit Format Types
-----------------
``pint`` comes with a variety of unit formats:

======= =============== ======================================================================
Spec    Name            Example
======= =============== ======================================================================
``D``   default         ``kilogram * meter / second ** 2``
``P``   pretty          ``kilogram·meter/second²``
``H``   HTML            ``kilogram meter/second<sup>2</sup>``
``L``   latex           ``\frac{\mathrm{kilogram} \cdot \mathrm{meter}}{\mathrm{second}^{2}}``
``Lx``  latex siunitx   ``\si[]{\kilo\gram\meter\per\second\squared}``
``C``   compact         ``kilogram*meter/second**2``
======= =============== ======================================================================

Custom Unit Format Types
------------------------
Using :py:func:`pint.register_unit_format`, it is possible to add custom
formats:

.. ipython::

   In [1]: u = ureg.Unit("m ** 3 / (s ** 2 * kg)")

   In [2]: import itertools
      ...:
      ...: @pint.register_unit_format("simple")
      ...: def format_unit_simple(unit, registry, **options):
      ...:     preformatted = [
      ...:         ("*" if p >= 0 else "/", n if abs(p) == 1 else f"{n} ** {abs(p)}")
      ...:         for n, p in unit.items()
      ...:     ]
      ...:     return " ".join(itertools.chain.from_iterable(
      ...:         [o, u] if i != 0 else [u]
      ...:         for i, (o, u) in enumerate(preformatted)
      ...:     ))

   In [3]: f"{u:~simple}"

where ``unit`` is a :py:class:`dict` subclass containing the unit names and
their exponents.

Quantity Format Specifications
------------------------------
The magnitude format is forwarded to the magnitude (for a unit-spec of ``H`` the
magnitude's ``_repr_html_`` is called).

Let's look at some more examples:

.. ipython:: python

    q = 1e-6 * u

    # modifiers
    f"{q:~P}"  # short pretty
    f"{q:~#P}"  # compact short pretty
    f"{q:P#~}"  # also compact short pretty

    # additional magnitude format
    f"{q:.2f~#P}"  # short compact pretty with 2 float digits
    f"{q:#~}"  # short compact default

Quantity Format Types
---------------------
There are no special quantity formats yet.

Modifiers
---------
======== =================================================== ================================
Modifier Meaning                                             Example
======== =================================================== ================================
``~``    Use the unit's symbol instead of its canonical name ``kg·m/s²`` (``f"{u:~P}"``)
``#``    Call :py:meth:`Quantity.to_compact` first           ``1.0 m·mg/s²`` (``f"{q:#~P}"``)
======== =================================================== ================================
