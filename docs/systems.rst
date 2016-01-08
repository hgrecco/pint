.. systems_:

Different Unit Systems (and default units)
==========================================

Pint Unit Registry has the concept of system, which is a group of units

    >>> import pint
    >>> ureg = pint.UnitRegistry(system='mks')
    >>> ureg.system
    'mks'

This has an effect in the base units. For example:

    >>> q = 3600. * ureg.meter / ureg.hour
    >>> q.to_base_units()
    <Quantity(1.0, 'meter / second')>

We can take a look for the available systems

    >>> ureg.systems
    frozenset({'US', 'mks', 'imperial', 'cgs'})

But if you change to cgs:

    >>> ureg.system = 'cgs'
    >>> q.to_base_units()
    <Quantity(100.0, 'centimeter / second')>

or more drastically to:

    >>> ureg.system = 'imperial'
    >>> '{:.3f}'.format(q.to_base_units())
    '1.094 yard / second'

..warning: In versions previous to 0.7 `to_base_units` returns quantities in the
           units of the definition files (which are called root units). For the definition file
           bundled with pint this is meter/gram/second. To get back this behaviour use `to_root_units`,
           set `ureg.system = None`


You can also use system to narrow down the list of compatible units:

    >>> ureg.system = 'mks'
    >>> ureg.get_compatible_units('meter')
    frozenset({<Unit('light_year')>, <Unit('angstrom')>, <Unit('US_survey_mile')>, <Unit('yard')>, <Unit('US_survey_foot')>, <Unit('US_survey_yard')>, <Unit('inch')>, <Unit('rod')>, <Unit('mile')>, <Unit('barleycorn')>, <Unit('foot')>, <Unit('mil')>})


    >>> ureg.system = 'imperial'
    >>> ureg.get_compatible_units('meter')
    frozenset({<Unit('US_survey_mile')>, <Unit('angstrom')>, <Unit('inch')>, <Unit('light_year')>, <Unit('barleycorn')>, <Unit('mile')>, <Unit('US_survey_foot')>, <Unit('rod')>, <Unit('US_survey_yard')>, <Unit('yard')>, <Unit('mil')>, <Unit('foot')>})

    >>> ureg.imperial.pint
    bla

    >>> ureg.us.pint
