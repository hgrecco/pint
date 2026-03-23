.. _quantity-to:

Unit Conversion Methods
=======================

Pint quantities provide several methods for converting to different units. This page
covers all ``to_*`` and ``ito_*`` methods available on a :class:`Quantity
<pint.quantity.Quantity>`.

The ``ito_*`` variants perform the conversion **in place** (modifying the quantity),
while the ``to_*`` variants return a **new** quantity and can be chained.

Setup used throughout this page:

.. doctest::

   >>> import pint
   >>> ureg = pint.UnitRegistry()
   >>> Q_ = ureg.Quantity


Summary Table
-------------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Method
     - ``Q_(5000, "m")``
     - ``Q_(200e-9, "s")``
     - ``Q_(1e6, "mW")``
     - ``Q_(1, "N*m")``
   * - ``.to_base_units()``
     - ``5000 m``
     - ``2e-07 s``
     - ``1000 kg·m²/s³``
     - ``1 kg·m²/s²``
   * - ``.to_root_units()``
     - ``5000 m``
     - ``2e-07 s``
     - ``1e6 g·m²/s³``
     - ``1000 g·m²/s²``
   * - ``.to_compact()``
     - ``5 km``
     - ``200 ns``
     - ``1 kW``
     - ``1 N·m``
   * - ``.to_reduced_units()``
     - ``5000 m``
     - ``2e-07 s``
     - ``1e6 mW``
     - ``1 N·m``
   * - ``.to_unprefixed()``
     - ``5000 m``
     - ``2e-07 s``
     - ``1000 W``
     - ``1 N·m``
   * - ``.to_base_units().to_compact()``
     - ``5 km``
     - ``200 ns``
     - ``1 Mg·m²/s³``
     - ``1 kg·m²/s²``
   * - ``.to_reduced_units().to_compact()``
     - ``5 km``
     - ``200 ns``
     - ``1 kW``
     - ``1 N·m``
   * - ``.to_unprefixed().to_compact()``
     - ``5 km``
     - ``200 ns``
     - ``1 kW``
     - ``1 N·m``
   * - ``.to_preferred([ureg.W, ureg.s, ureg.J, ureg.m])`` †
     - ``5000 m``
     - ``2e-07 s``
     - ``1 kW`` †
     - ``1 J`` †

.. note::

   † ``.to_preferred()`` requires ``scipy`` for multi-dimensional quantities.
   Values marked † are expected results when scipy is available.

.. note::

   ``.to_base_units().to_compact()`` on a prefixed unit like ``mW`` expands to
   base SI units (``kg·m²/s³``) before rescaling — the result (``Mg·m²/s³``) is
   numerically correct but less readable. Use ``.to_reduced_units().to_compact()``
   or ``.to_unprefixed().to_compact()`` to stay in named units like ``kW``.

.. note::

   ``.to_preferred()`` requires ``scipy`` and is not shown in the table above.
   See :ref:`to_preferred_section` below.


to / ito
--------

Convert to specific units by passing a unit string, unit object, or another quantity.

.. doctest::

   >>> distance = 1000 * ureg.meter
   >>> distance.to("km")
   <Quantity(1.0, 'kilometer')>
   >>> speed = Q_(60, "miles/hour")
   >>> speed.to("km/hour")
   <Quantity(96.56064, 'kilometer / hour')>

``ito`` modifies in place:

.. doctest::

   >>> distance = 1000 * ureg.meter
   >>> distance.ito("km")
   >>> distance
   <Quantity(1.0, 'kilometer')>


to_base_units / ito_base_units
-------------------------------

Convert to base units as defined by the unit registry (SI base units by default).

.. doctest::

   >>> Q_(1, "km").to_base_units()
   <Quantity(1000.0, 'meter')>
   >>> Q_(1, "hour").to_base_units()
   <Quantity(3600.0, 'second')>
   >>> Q_(1, "newton").to_base_units()
   <Quantity(1.0, 'kilogram * meter / second ** 2')>


to_root_units / ito_root_units
-------------------------------

Convert to root units — the primitive units before any system-level
transformations. For most units this matches ``to_base_units``, but differs for
units like ``mW`` where the root unit uses grams rather than kilograms.

.. doctest::

   >>> Q_(1, "km").to_root_units()
   <Quantity(1000.0, 'meter')>
   >>> Q_(1, "newton").to_root_units()
   <Quantity(1.0, 'kilogram * meter / second ** 2')>


to_reduced_units / ito_reduced_units
--------------------------------------

Reduce to one unit per dimension, cancelling units where possible. This does not
reduce compound units like ``J/kg`` to ``m²/s²``.

.. doctest::

   >>> Q_(1, "m * km").to_reduced_units()
   <Quantity(1000.0, 'meter ** 2')>
   >>> Q_(1, "m/km").to_reduced_units()
   <Quantity(0.001, '')>


to_compact / ito_compact
-------------------------

Rescale to the most human-readable SI-prefixed unit for the magnitude.

.. doctest::

   >>> Q_(0.00042, "m").to_compact()
   <Quantity(0.42, 'millimeter')>
   >>> Q_(12000, "m").to_compact()
   <Quantity(12.0, 'kilometer')>
   >>> Q_(5e-9, "s").to_compact()
   <Quantity(5.0, 'nanosecond')>

Pass a unit to compact within a specific unit family:

.. doctest::

   >>> Q_(0.01, "kg * m / s**2").to_compact("N")
   <Quantity(10.0, 'millinewton')>


.. _to_preferred_section:

to_preferred / ito_preferred
-----------------------------

Convert to a unit composed of a given list of preferred units. Requires ``scipy``.
If no preferred units are given, the registry's ``default_preferred_units`` are used.

.. doctest::

   >>> Q_(1, "acre").to_preferred([ureg.meter])
   <Quantity(4046.87261, 'meter ** 2')>
   >>> Q_(1, "km/hour").to_preferred([ureg.mile])
   <Quantity(0.62137119, 'mile / hour')>
   >>> Q_(4.184, "J").to_preferred([ureg.W, ureg.s])
   <Quantity(4.184, 'watt * second')>


to_unprefixed / ito_unprefixed
--------------------------------

Remove SI prefixes without converting to base units. Useful when you want the
unprefixed form of a unit while staying in the same unit family.

.. doctest::

   >>> Q_(1, "km").to_unprefixed()
   <Quantity(1000.0, 'meter')>
   >>> Q_(1, "ms").to_unprefixed()
   <Quantity(0.001, 'second')>
   >>> Q_(1, "MW").to_unprefixed()
   <Quantity(1000000.0, 'watt')>


to_tuple
--------

Serialize a quantity to a ``(magnitude, unit_items)`` tuple. Can be reconstructed
with :meth:`Quantity.from_tuple`.

.. doctest::

   >>> q = Q_(1.5, "m/s")
   >>> tup = q.to_tuple()
   >>> tup
   (1.5, (('meter', 1), ('second', -1)))
   >>> Q_.from_tuple(tup)
   <Quantity(1.5, 'meter / second')>


to_timedelta
------------

Convert a time quantity to a :class:`datetime.timedelta` object.

.. doctest::

   >>> import datetime
   >>> Q_(2.5, "hours").to_timedelta()
   datetime.timedelta(seconds=9000)
   >>> Q_(500, "ms").to_timedelta()
   datetime.timedelta(microseconds=500000)


Chaining Conversions
--------------------

Since ``to_*`` methods return new quantities, they can be chained. The ``ito_*``
variants return ``None`` and cannot be chained.

.. doctest::

   >>> Q_(1, "m * km").to_reduced_units().to_compact()
   <Quantity(1.0, 'kilometer ** 2')>
   >>> Q_(1, "Wh").to_base_units().to_compact()
   <Quantity(3.6, 'kilojoule')>
   >>> Q_(0.003, "km").to_unprefixed().to_compact()
   <Quantity(3.0, 'meter')>
