"""
pint.babel
~~~~~~~~~~

:copyright: 2016 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from .compat import HAS_BABEL

# Canonical pint unit names mapped to CLDR unit ids, see
# https://github.com/unicode-org/cldr/blob/main/common/validity/unit.xml
_babel_units: dict[str, str] = dict(
    acre="area-acre",
    acre_foot="volume-acre-foot",
    ampere="electric-ampere",
    arcminute="angle-arc-minute",
    arcsecond="angle-arc-second",
    astronomical_unit="length-astronomical-unit",
    bit="digital-bit",
    british_thermal_unit="energy-british-thermal-unit",
    byte="digital-byte",
    calorie="energy-calorie",
    carat="mass-carat",
    centiliter="volume-centiliter",
    centimeter="length-centimeter",
    century="duration-century",
    cubic_centimeter="volume-cubic-centimeter",
    cubic_foot="volume-cubic-foot",
    cubic_inch="volume-cubic-inch",
    cubic_yard="volume-cubic-yard",
    cup="volume-cup",
    day="duration-day",
    deciliter="volume-deciliter",
    decimeter="length-decimeter",
    degree="angle-degree",
    degree_Celsius="temperature-celsius",
    degree_Fahrenheit="temperature-fahrenheit",
    electron_volt="energy-electronvolt",
    fluid_ounce="volume-fluid-ounce",
    foot="length-foot",
    gallon="volume-gallon",
    gigahertz="frequency-gigahertz",
    gigawatt="power-gigawatt",
    gram="mass-gram",
    hectare="area-hectare",
    hectoliter="volume-hectoliter",
    hectopascal="pressure-hectopascal",
    hertz="frequency-hertz",
    horsepower="power-horsepower",
    hour="duration-hour",
    inch="length-inch",
    inch_Hg="pressure-inch-ofhg",
    joule="energy-joule",
    kelvin="temperature-kelvin",
    kilocalorie="energy-kilocalorie",
    kilogram="mass-kilogram",
    kilohertz="frequency-kilohertz",
    kilojoule="energy-kilojoule",
    kilometer="length-kilometer",
    kilometer_per_hour="speed-kilometer-per-hour",
    kilowatt="power-kilowatt",
    kilowatt_hour="energy-kilowatt-hour",
    knot="speed-knot",
    light_year="length-light-year",
    liter="volume-liter",
    lux="light-lux",
    megahertz="frequency-megahertz",
    megaliter="volume-megaliter",
    megawatt="power-megawatt",
    meter="length-meter",
    meter_per_second="speed-meter-per-second",
    meter_per_second_squared="acceleration-meter-per-square-second",
    metric_ton="mass-tonne",
    microgram="mass-microgram",
    micrometer="length-micrometer",
    microsecond="duration-microsecond",
    mile="length-mile",
    mile_per_hour="speed-mile-per-hour",
    milliampere="electric-milliampere",
    millibar="pressure-millibar",
    milligram="mass-milligram",
    milliliter="volume-milliliter",
    millimeter="length-millimeter",
    millimeter_Hg="pressure-millimeter-ofhg",
    millisecond="duration-millisecond",
    milliwatt="power-milliwatt",
    minute="duration-minute",
    month="duration-month",
    nanometer="length-nanometer",
    nanosecond="duration-nanosecond",
    nautical_mile="length-nautical-mile",
    ohm="electric-ohm",
    ounce="mass-ounce",
    parsec="length-parsec",
    picometer="length-picometer",
    pint="volume-pint",
    pound="mass-pound",
    pound_force_per_square_inch="pressure-pound-force-per-square-inch",
    quart="volume-quart",
    radian="angle-radian",
    second="duration-second",
    square_foot="area-square-foot",
    square_inch="area-square-inch",
    square_mile="area-square-mile",
    square_yard="area-square-yard",
    standard_gravity="acceleration-g-force",
    tablespoon="volume-tablespoon",
    teaspoon="volume-teaspoon",
    ton="mass-ton",
    troy_ounce="mass-ounce-troy",
    turn="angle-revolution",
    volt="electric-volt",
    watt="power-watt",
    week="duration-week",
    yard="length-yard",
    year="duration-year",
)

# Deprecated CLDR ids still used by older babel releases.
_babel_units_deprecated: dict[str, str] = {
    "acceleration-meter-per-square-second": "acceleration-meter-per-second-squared",
    "mass-tonne": "mass-metric-ton",
    "pressure-inch-ofhg": "pressure-inch-hg",
    "pressure-millimeter-ofhg": "pressure-millimeter-of-mercury",
    "pressure-pound-force-per-square-inch": "pressure-pound-per-square-inch",
}

if not HAS_BABEL:
    _babel_units = {}
    _babel_units_deprecated = {}

_babel_systems: dict[str, str] = dict(mks="metric", imperial="uksystem", US="ussystem")

_babel_lengths: list[str] = ["narrow", "short", "long"]
