.. _systems:

Different Unit Systems (and default units)
==========================================

Pint Unit Registry has the concept of system, which is a group of units

.. doctest::

    >>> import pint
    >>> ureg = pint.UnitRegistry(system='mks')
    >>> ureg.default_system
    'mks'

This has an effect in the base units. For example:

.. doctest::

    >>> q = 3600. * ureg.meter / ureg.hour
    >>> q.to_base_units()
    <Quantity(1.0, 'meter / second')>

But if you change to cgs:

.. doctest::

    >>> ureg.default_system = 'cgs'
    >>> q.to_base_units()
    <Quantity(100.0, 'centimeter / second')>

or more drastically to:

.. doctest::

    >>> ureg.default_system = 'imperial'
    >>> '{:.3f}'.format(q.to_base_units())
    '1.094 yard / second'

.. warning:: In versions previous to 0.7, ``to_base_units()`` returns quantities in the
             units of the definition files (which are called root units). For the definition file
             bundled with pint this is meter/gram/second. To get back this behaviour use ``to_root_units()``,
             set ``ureg.system = None``

You can check which unit systems are available:

.. doctest::

    >>> dir(ureg.sys)
    ['Planck', 'SI', 'US', 'atomic', 'cgs', 'imperial', 'mks']

Or which units are available within a particular system:

.. doctest::

    >>> dir(ureg.sys.imperial)
    ['UK_force_ton', 'UK_hundredweight', ... 'cubic_foot', 'cubic_inch', ... 'thou', 'ton', 'yard']

Notice that this give you the opportunity to choose within units with colliding names:

.. doctest::

    >>> (1 * ureg.sys.imperial.pint).to('liter')
    <Quantity(0.568261..., 'liter')>
    >>> (1 * ureg.sys.US.pint).to('liter')
    <Quantity(0.473176..., 'liter')>
    >>> (1 * ureg.sys.US.pint).to(ureg.sys.imperial.pint)
    <Quantity(0.832674..., 'imperial_pint')>
