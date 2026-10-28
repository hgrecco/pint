"""Static check of pint against the metrology API protocols.

Needs metrology_apis, which isn't on PyPI yet. From the repository root:

    pixi run -e typecheck pip install --no-deps git+https://github.com/quantity-dev/metrology-apis@25f975f8b4193d5bfca6d49084c68c14ccb27abd
    pixi run -e typecheck pyright -p pint/testsuite/typing/metrology

0 errors means every pint object below conforms to its protocol.
"""

from typing import Any

import metrology_apis as mapis

from pint import UnitRegistry
from pint.util import UnitsContainer

ureg = UnitRegistry()
dimension = UnitsContainer({"[length]": 1})
unit = ureg.Unit("m")
quantity = ureg.Quantity(1.0, "m")

mn: mapis.MetrologyNamespace = ureg.__metrology_namespace__()
has_namespace: mapis.HasMetrologyNamespace = ureg
full_dimension: mapis.Dimension = dimension
full_unit: mapis.Unit = unit
full_quantity: mapis.Quantity = quantity
float_quantity: mapis.Quantity[float, Any, Any] = quantity
