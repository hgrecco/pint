.. _default-units:

List of default units
======================

Pint ships with a comprehensive list of physical units, prefixes and
dimensions, defined in a plain text file that you can browse directly on
GitHub:

`default_en.txt <https://github.com/hgrecco/pint/blob/master/pint/default_en.txt>`_

Any unit defined there is available as an attribute of a registry, and as a
name you can use when parsing a unit string:

.. code-block:: python

    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> ureg.meter
    Unit("meter")
    >>> ureg("2 km/hour")
    Quantity(2.0, "kilometer / hour")

You can also list every unit name known to a registry from Python, without
opening the definition file:

.. code-block:: python

    >>> units = sorted(ureg)
    >>> "meter" in units
    True

If you need a unit that isn't defined, or want to change the definition of an
existing one, see :ref:`defining`.
