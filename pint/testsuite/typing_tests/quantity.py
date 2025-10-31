from typing import reveal_type
import pint


ureg = pint.UnitRegistry() # type: pint.UnitRegistry
a = ureg.Quantity(10, "meter")
b = 5 * ureg.meter
c =  ureg.meter * 5
reveal_type(a)
reveal_type(b)
reveal_type(c)



position_vector1: pint.Quantity[list[float]] = ureg.m * [1, 2, 3]
position_vector2: pint.Quantity[list[float]] = [1, 2, 3] * ureg.m
