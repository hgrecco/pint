"""
    pint.delegates.formatter.plain
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implements plain text formatters:
    - Raw: as simple as it gets (no locale aware, no unit formatter.)
    - Default: used when no string spec is given.
    - Compact: like default but with less spaces.
    - Pretty: pretty printed formatter.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import re
from ...compat import ndarray, np, Unpack
from ._spec_helpers import (
    pretty_fmt_exponent,
    split_format,
    join_mu,
    join_unc,
    remove_custom_flags,
)

from ..._typing import Magnitude

from ._format_helpers import format_compound_unit, BabelKwds, formatter, override_locale

if TYPE_CHECKING:
    from ...facets.plain import PlainQuantity, PlainUnit, MagnitudeT
    from ...facets.measurement import Measurement


_EXP_PATTERN = re.compile(r"([0-9]\.?[0-9]*)e(-?)\+?0*([0-9]+)")


class DefaultFormatter:
    """Simple, localizable plain text formatter.

    A formatter is a class with methods to format into string each of the objects
    that appear in pint (magnitude, unit, quantity, uncertainty, measurement)
    """

    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        """Format scalar/array into string
        given a string formatting specification and locale related arguments.
        """
        with override_locale(mspec, babel_kwds.get("locale", None)) as format_number:
            if isinstance(magnitude, ndarray) and magnitude.ndim > 0:
                # Use custom ndarray text formatting--need to handle scalars differently
                # since they don't respond to printoptions
                with np.printoptions(formatter={"float_kind": format_number}):
                    mstr = format(magnitude).replace("\n", "")
            else:
                mstr = format_number(magnitude)

        return mstr

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)
        """Format a unit (can be compound) into string
        given a string formatting specification and locale related arguments.
        """

        return formatter(
            units,
            as_ratio=True,
            single_denominator=False,
            product_fmt=" * ",
            division_fmt=" / ",
            power_fmt="{} ** {}",
            parentheses_fmt=r"({})",
        )

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        qspec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        """Format a quantity (magnitude and unit) into string
        given a string formatting specification and locale related arguments.
        """

        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            qspec, registry.formatter.default_format, registry.separate_format_defaults
        )

        joint_fstring = "{} {}"
        return join_mu(
            joint_fstring,
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )

    def format_uncertainty(
        self,
        uncertainty,
        unc_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        """Format an uncertainty magnitude (nominal value and stdev) into string
        given a string formatting specification and locale related arguments.
        """

        return format(uncertainty, unc_spec).replace("+/-", " +/- ")

    def format_measurement(
        self,
        measurement: Measurement,
        meas_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        """Format an measurement (uncertainty and units) into string
        given a string formatting specification and locale related arguments.
        """

        registry = measurement._REGISTRY

        mspec, uspec = split_format(
            meas_spec,
            registry.formatter.default_format,
            registry.separate_format_defaults,
        )

        unc_spec = remove_custom_flags(meas_spec)

        joint_fstring = "{} {}"

        return join_unc(
            joint_fstring,
            "(",
            ")",
            self.format_uncertainty(measurement.magnitude, unc_spec, **babel_kwds),
            self.format_unit(measurement.units, uspec, **babel_kwds),
        )


class CompactFormatter:
    """Simple, localizable plain text formatter without extra spaces."""

    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        with override_locale(mspec, babel_kwds.get("locale", None)) as format_number:
            if isinstance(magnitude, ndarray) and magnitude.ndim > 0:
                # Use custom ndarray text formatting--need to handle scalars differently
                # since they don't respond to printoptions
                with np.printoptions(formatter={"float_kind": format_number}):
                    mstr = format(magnitude).replace("\n", "")
            else:
                mstr = format_number(magnitude)

        return mstr

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)

        return formatter(
            units,
            as_ratio=True,
            single_denominator=False,
            product_fmt="*",  # TODO: Should this just be ''?
            division_fmt="/",
            power_fmt="{}**{}",
            parentheses_fmt=r"({})",
        )

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        qspec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            qspec, registry.formatter.default_format, registry.separate_format_defaults
        )

        joint_fstring = "{} {}"

        return join_mu(
            joint_fstring,
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )

    def format_uncertainty(
        self,
        uncertainty,
        unc_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        return format(uncertainty, unc_spec).replace("+/-", "+/-")

    def format_measurement(
        self,
        measurement: Measurement,
        meas_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = measurement._REGISTRY

        mspec, uspec = split_format(
            meas_spec,
            registry.formatter.default_format,
            registry.separate_format_defaults,
        )

        unc_spec = remove_custom_flags(meas_spec)

        joint_fstring = "{} {}"

        return join_unc(
            joint_fstring,
            "(",
            ")",
            self.format_uncertainty(measurement.magnitude, unc_spec, **babel_kwds),
            self.format_unit(measurement.units, uspec, **babel_kwds),
        )


class PrettyFormatter:
    """Pretty printed localizable plain text formatter without extra spaces."""

    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        with override_locale(mspec, babel_kwds.get("locale", None)) as format_number:
            if isinstance(magnitude, ndarray) and magnitude.ndim > 0:
                # Use custom ndarray text formatting--need to handle scalars differently
                # since they don't respond to printoptions
                with np.printoptions(formatter={"float_kind": format_number}):
                    mstr = format(magnitude).replace("\n", "")
            else:
                mstr = format_number(magnitude)

            m = _EXP_PATTERN.match(mstr)

            if m:
                exp = int(m.group(2) + m.group(3))
                mstr = _EXP_PATTERN.sub(r"\1×10" + pretty_fmt_exponent(exp), mstr)

            return mstr

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)

        return formatter(
            units,
            as_ratio=True,
            single_denominator=False,
            product_fmt="·",
            division_fmt="/",
            power_fmt="{}{}",
            parentheses_fmt="({})",
            exp_call=pretty_fmt_exponent,
            sort_func=lambda x: unit._REGISTRY.formatter.default_sort_func(
                x, unit._REGISTRY
            ),
        )

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        qspec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            qspec, registry.formatter.default_format, registry.separate_format_defaults
        )

        joint_fstring = "{} {}"

        return join_mu(
            joint_fstring,
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )

    def format_uncertainty(
        self,
        uncertainty,
        unc_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        return format(uncertainty, unc_spec).replace("±", " ± ")

    def format_measurement(
        self,
        measurement: Measurement,
        meas_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = measurement._REGISTRY

        mspec, uspec = split_format(
            meas_spec,
            registry.formatter.default_format,
            registry.separate_format_defaults,
        )

        unc_spec = meas_spec
        joint_fstring = "{} {}"

        return join_unc(
            joint_fstring,
            "(",
            ")",
            self.format_uncertainty(measurement.magnitude, unc_spec, **babel_kwds),
            self.format_unit(measurement.units, uspec, **babel_kwds),
        )


class RawFormatter:
    """Very simple non-localizable plain text formatter.

    Ignores all pint custom string formatting specification.
    """

    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        return str(magnitude)

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)

        return " * ".join(k if v == 1 else f"{k} ** {v}" for k, v in units)

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        qspec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            qspec, registry.formatter.default_format, registry.separate_format_defaults
        )

        joint_fstring = "{} {}"
        return join_mu(
            joint_fstring,
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )

    def format_uncertainty(
        self,
        uncertainty,
        unc_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        return format(uncertainty, unc_spec)

    def format_measurement(
        self,
        measurement: Measurement,
        meas_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = measurement._REGISTRY

        mspec, uspec = split_format(
            meas_spec,
            registry.formatter.default_format,
            registry.separate_format_defaults,
        )

        unc_spec = remove_custom_flags(meas_spec)

        joint_fstring = "{} {}"

        return join_unc(
            joint_fstring,
            "(",
            ")",
            self.format_uncertainty(measurement.magnitude, unc_spec, **babel_kwds),
            self.format_unit(measurement.units, uspec, **babel_kwds),
        )
