.. _default-units:

List of default units
======================

The default definitions can be found at `default_en.txt
<https://github.com/hgrecco/pint/blob/master/pint/default_en.txt>`_ and have
been used to generate the following lists.

Hover over the name of a unit to see its value.

Prefixes
--------

%%PINT_PREFIXES%%

Units
-----

%%PINT_UNITS%%

Unit systems
------------

A system is a named subset of units. Setting ``ureg.system = "imperial"``
(for example) changes which unit ``to_base_units()`` converts to. See
:ref:`systems` for more.

%%PINT_SYSTEMS%%

Groups
------

A group is a named subset of units, similar to a system.

%%PINT_GROUPS%%

Available contexts
-------------------

A context lets you convert between otherwise incompatible dimensions (e.g.
wavelength and frequency) within a ``with ureg.context(...):`` block. See
:doc:`user/contexts` for more.

%%PINT_CONTEXTS%%
