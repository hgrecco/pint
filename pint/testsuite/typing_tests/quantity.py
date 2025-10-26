import pint


ureg = pint.UnitRegistry() # type: pint.UnitRegistry
position_vector: pint.Quantity[list[float]] = ureg.m * [1, 2, 3]
# position_vector: pint.Quantity[list[float]] = [1, 2, 3] * ureg.m
