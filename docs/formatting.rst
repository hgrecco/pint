.. _formatting:
.. currentmodule:: pint


.. ipython:: python
   :suppress:

   import pint


String formatting
=================
Unit Format Specifications
--------------------------
A :doc:`format specification <formatspec>` used to format :py:class:`Unit` objects in
e.g. f-strings consists of a ``type`` and optionally a "modifier", where the order does
not matter. For example, ``P~`` and ``~P`` have the same effect. If the ``type`` is
omitted, or when using the :py:class:`str` builtin, the object's
:py:attr:`Unit.default_format` property is used, falling back to
:py:attr:`UnitRegistry.default_format` if that is not set. If both are unset, the
default format (``"D"``) is used.

.. note::

   Modifiers may still be used without specifying the type: ``"~"`` is a valid format
   specification.

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
   f"{u}"  # default: compact
   u.default_format = "P"
   f"{u}"  # default: pretty
   u.default_format = ""  # TODO: switch to None
   ureg.default_format = ""  # TODO: switch to None
   f"{u}"  # default: default

**TODO**: describe quantity formats

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

Quantity Formats
----------------
.. ipython:: python

    q = 1e-6 * u

    # modifiers
    f"{q:~P}"  # short pretty
    f"{q:~#P}"  # compact short pretty
    f"{q:P#~}"  # also compact short pretty

    # additional magnitude format
    f"{q:.2f~#P}"  # short compact pretty with 8 float digits
    f"{q:#~}"  # compact short default

Modifiers
---------
======== =================================================== ================================
Modifier Meaning                                             Example
======== =================================================== ================================
``~``    Use the unit's symbol instead of its canonical name ``kg·m/s²`` (``f"{u:~P}"``)
``#``    Call :py:meth:`Quantity.to_compact` first           ``1.0 m·mg/s²`` (``f"{q:#~P}"``)
======== =================================================== ================================
