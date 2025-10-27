import pint


ureg = pint.UnitRegistry() # type: pint.UnitRegistry
a = ureg.Quantity(10, "meter")
b: pint.Quantity = 5 * ureg.meter

position_vector1: pint.Quantity[list[float]] = ureg.m * [1, 2, 3]
position_vector2: pint.Quantity[list[float]] = [1, 2, 3] * ureg.m

a.
b.