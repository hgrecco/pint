"""
    pint.formatter
    ~~~~~~~~~~~~~~

    Format units for pint.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations


# Backwards compatiblity stuff
from .delegates.formatter.latex import (
    vector_to_latex,  # noqa: F401
    matrix_to_latex,  # noqa: F401
    ndarray_to_latex_parts,  # noqa: F401
    ndarray_to_latex,  # noqa: F401
    latex_escape,  # noqa: F401
    siunitx_format_unit,  # noqa: F401
    _EXP_PATTERN,  # noqa: F401
)  # noqa
from .delegates.formatter._spec_helpers import (
    FORMATTER,  # noqa: F401
    _BASIC_TYPES,  # noqa: F401
    parse_spec as _parse_spec,  # noqa: F401
    _join,  # noqa: F401
    _PRETTY_EXPONENTS,  # noqa: F401
    pretty_fmt_exponent as _pretty_fmt_exponent,  # noqa: F401
    extract_custom_flags,  # noqa: F401
    remove_custom_flags,  # noqa: F401
    split_format,  # noqa: F401
    REGISTERED_FORMATTERS,
)  # noqa
from .delegates.formatter._to_register import register_unit_format  # noqa: F401


def format_unit(unit, spec: str, registry=None, **options):
    # registry may be None to allow formatting `UnitsContainer` objects
    # in that case, the spec may not be "Lx"

    if not unit:
        if spec.endswith("%"):
            return ""
        else:
            return "dimensionless"

    if not spec:
        spec = "D"

    if registry is None:
        _formatter = REGISTERED_FORMATTERS.get(spec, None)
    else:
        try:
            _formatter = registry._formatters[spec]
        except Exception:
            _formatter = registry._formatters.get(spec, None)

    if _formatter is None:
        raise ValueError(f"Unknown conversion specified: {spec}")

    return _formatter.format_unit(unit)
