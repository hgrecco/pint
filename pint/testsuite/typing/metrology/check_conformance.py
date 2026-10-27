"""Static check of pint against the metrology API protocols.

Needs metrology_apis, which isn't on PyPI yet. From the repository root:

    pixi run -e typecheck pip install --no-deps git+https://github.com/quantity-dev/metrology-apis@328e95e10a7673aded0b8b3dd524213d85ae2702
    pixi run -e typecheck pyright -p pint/testsuite/typing/metrology

0 errors means every line below behaves as labelled:
- plain assignments conform;
- lines with ``# pyright: ignore`` are known failures. Since
  ``reportUnnecessaryTypeIgnoreComment`` is on, pyright errors if one of
  them starts passing.

The per-member protocols are copied from ``metrology_apis`` (spec version
0.0.1.dev0) so each member can be checked on its own.
"""

from typing import Any, Protocol, Self, override

import metrology_apis as mapis
import optype as op

from pint import UnitRegistry
from pint.util import UnitsContainer

ureg = UnitRegistry()
dimension = UnitsContainer({"[length]": 1})
unit = ureg.Unit("m")
quantity = ureg.Quantity(1.0, "m")


# ===================================================================
# Dimension: every member except __metrology_namespace__


class DimensionOps(Protocol):
    def __mul__(self, other: Self, /) -> Self: ...
    def __truediv__(self, other: Self, /) -> Self: ...
    def __pow__(self, other: int, /) -> Self: ...
    def __rmul__(self, other: Self, /) -> Self: ...
    def __rtruediv__(self, other: Self, /) -> Self: ...


dimension_ops: DimensionOps = dimension


# ===================================================================
# Unit: every member except __metrology_namespace__


class UnitOps(Protocol):
    @property
    def dimension(self) -> UnitsContainer: ...
    def __mul__(self, other: Self, /) -> Self: ...
    def __truediv__(self, other: Self, /) -> Self: ...
    def __pow__(self, other: int | float, /) -> Self: ...
    def __rmul__(self, other: Self, /) -> Self: ...
    def __rtruediv__(self, other: Self, /) -> Self: ...


unit_ops: UnitOps = unit


# ===================================================================
# Quantity: one protocol per member, because the spec's signatures refer
# to the protocol itself


class QValue[V, U, D](Protocol):
    @property
    def value(self) -> V: ...


class QUnit[V, U, D](Protocol):
    @property
    def unit(self) -> U: ...


class QLt[V, U, D](Protocol):
    def __lt__[B](self: "QLt[V, U, D]", other: "QLt[op.CanLt[V, B], U, D]", /) -> B: ...


class QLe[V, U, D](Protocol):
    def __le__[B](self: "QLe[V, U, D]", other: "QLe[op.CanLe[V, B], U, D]", /) -> B: ...


class QGt[V, U, D](Protocol):
    def __gt__[B](self: "QGt[V, U, D]", other: "QGt[op.CanGt[V, B], U, D]", /) -> B: ...


class QGe[V, U, D](Protocol):
    def __ge__[B](self: "QGe[V, U, D]", other: "QGe[op.CanGe[V, B], U, D]", /) -> B: ...


class QEq[V, U, D](Protocol):
    @override
    def __eq__[B](  # pyright: ignore[reportIncompatibleMethodOverride]
        self: "QEq[V, U, D]", other: "QEq[op.CanEq[V, B], U, D]", /
    ) -> B: ...


class QAdd[V, U, D](Protocol):
    def __add__[R](
        self: "QAdd[V, U, D]", other: "QAdd[op.CanRAdd[V, R], U, D]", /
    ) -> "QAdd[R, U, D]": ...


class QNeg[V, U, D](Protocol):
    def __neg__[R](self: "QNeg[op.CanNeg[R], U, D]") -> "QNeg[R, U, D]": ...


q_value: QValue[float, Any, Any] = quantity
q_unit: QUnit[float, Any, Any] = quantity
q_lt: QLt[float, Any, Any] = quantity
q_le: QLe[float, Any, Any] = quantity
q_gt: QGt[float, Any, Any] = quantity
q_ge: QGe[float, Any, Any] = quantity

# Known failures. These signatures can't be satisfied under pyright even by
# a class that copies them word for word, so they show a spec problem, not
# a pint one. ==, !=, +x, abs(), -, *, /, ** and the reflected operators
# fail the same way.
q_eq: QEq[float, Any, Any] = quantity  # pyright: ignore[reportAssignmentType]
q_add: QAdd[float, Any, Any] = quantity  # pyright: ignore[reportAssignmentType]
q_neg: QNeg[float, Any, Any] = quantity  # pyright: ignore[reportAssignmentType]


# ===================================================================
# Full spec protocols: known failures. Every __metrology_namespace__ needs
# asquantity() to return a spec Quantity, which nothing satisfies (above).

full_namespace: mapis.MetrologyNamespace = ureg.__metrology_namespace__()  # pyright: ignore[reportAssignmentType]
full_unit: mapis.Unit = unit  # pyright: ignore[reportAssignmentType]
full_quantity: mapis.Quantity = quantity  # pyright: ignore[reportAssignmentType]
