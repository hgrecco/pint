.. _nonmult:


Temperature conversion
======================

This is not supported by most

    >>> home = 25.4 * ureg.degC
    >>> print(home.to('degF'))


For every temperature unit in the registry, there is also a *delta* counterpart
to specify differences. For example, the

    >>> increase = 12.3 * ureg.delta_degC
    >>> print(increase.to(ureg.delta_degK))
    12.3
    >>> print(increase.to(ureg.delta_degF))

which is different from:

    >>> print((12.3 * ureg.degC).to(ureg.delta_degK)
    >>> print((12.3 * ureg.degC).to(ureg.delta_degC)

Subtraction of two temperatures also yields a *delta* unit.

    >>> 25.4 * ureg.degC - 10. * ureg.degC
    15.4 delta_degC

Differences in temperature are multiplicative:

The parser knows about *delta* units and use them when a temperature unit is found
in a multiplicative context. For example, here:

    >>> ureg.parse_units('degC/meter')
    delta_degC / meter

but not here:

    >>> ureg.parse_units('degC')
    degC

You can override this behaviour

    >>> ureg.parse_units('degC/meter', to_delta=False)
    degC / meter

