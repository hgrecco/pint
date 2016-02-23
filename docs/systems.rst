.. _systems:

Different Unit Systems (and default units)
==========================================

Pint Unit Registry has the concept of system, which is a group of units

    >>> import pint
    >>> ureg = pint.UnitRegistry(system='mks')
    >>> ureg.default_system
    'mks'

This has an effect in the base units. For example:

    >>> q = 3600. * ureg.meter / ureg.hour
    >>> q.to_base_units()
    <Quantity(1.0, 'meter / second')>

But if you change to cgs:

    >>> ureg.default_system = 'cgs'
    >>> q.to_base_units()
    <Quantity(100.0, 'centimeter / second')>

or more drastically to:

    >>> ureg.default_system = 'imperial'
    >>> '{:.3f}'.format(q.to_base_units())
    '1.094 yard / second'

..warning: In versions previous to 0.7 `to_base_units` returns quantities in the
           units of the definition files (which are called root units). For the definition file
           bundled with pint this is meter/gram/second. To get back this behaviour use `to_root_units`,
           set `ureg.system = None`


You can also use system to narrow down the list of compatible units:

    >>> ureg.default_system = 'mks'
    >>> ureg.get_compatible_units('meter')
    frozenset({<Unit('light_year')>, <Unit('angstrom')>})

or for imperial units:

    >>> ureg.default_system = 'imperial'
    >>> ureg.get_compatible_units('meter')
    frozenset({<Unit('thou')>, <Unit('league')>, <Unit('nautical_mile')>, <Unit('inch')>, <Unit('mile')>, <Unit('yard')>, <Unit('foot')>})


You can check which unit systems are available:

    >>> dir(ureg.sys)
    ['US', 'cgs', 'imperial', 'mks']

Or which units are available within a particular system:

    >>> dir(ureg.sys.imperial)
    ['UK_hundredweight', 'UK_ton', 'acre_foot', 'cubic_foot', 'cubic_inch', 'cubic_yard', 'drachm', 'foot', 'grain', 'imperial_barrel', 'imperial_bushel', 'imperial_cup', 'imperial_fluid_drachm', 'imperial_fluid_ounce', 'imperial_gallon', 'imperial_gill', 'imperial_peck', 'imperial_pint', 'imperial_quart', 'inch', 'long_hunderweight', 'long_ton', 'mile', 'ounce', 'pound', 'quarter', 'short_hunderdweight', 'short_ton', 'square_foot', 'square_inch', 'square_mile', 'square_yard', 'stone', 'yard']

Notice that this give you the opportunity to choose within units with colliding names:

    >>> (1 * ureg.sys.imperial.pint).to('liter')
    <Quantity(0.5682612500000002, 'liter')>
    >>> (1 * ureg.sys.US.pint).to('liter')
    <Quantity(0.47317647300000004, 'liter')>
    >>> (1 * ureg.sys.US.pint).to(ureg.sys.imperial.pint)
    <Quantity(0.8326741846289889, 'imperial_pint')>
