import pint

ureg = pint.UnitRegistry()
UC = pint.util.UnitsContainer
q=ureg.Quantity("0.1234 s")
units = [ureg.Unit(u) for u in ["us", "ms", "s", "min", "hour", "day"]]
q.to_human({UC({'[time]':1}): units})

ureg = pint.UnitRegistry()
UC = pint.util.UnitsContainer
q=ureg.Quantity("0.1234 m**3")
units = [ureg.Unit(u) for u in ["us", "ms", "s", "min", "hour", "day"]]
units2 = [ureg.Unit(u) for u in ["m**3", "liter",]]
q.to_human({UC({'[time]':1}): units, UC({'[length]':3}): units2})
