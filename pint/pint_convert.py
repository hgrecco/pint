#!/usr/bin/env python3

"""
    pint-convert
    ~~~~~~~~~~~~

    :copyright: 2020 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import argparse
import contextlib
import re

from pint import UnitRegistry

parser = argparse.ArgumentParser(description="Unit converter.", usage=argparse.SUPPRESS)
parser.add_argument(
    "-s",
    "--system",
    metavar="sys",
    default="SI",
    help="unit system to convert to (default: SI)",
)
parser.add_argument(
    "-p",
    "--prec",
    metavar="n",
    type=int,
    default=12,
    help="number of maximum significant figures (default: 12)",
)
parser.add_argument(
    "-u",
    "--prec-unc",
    metavar="n",
    type=int,
    default=2,
    help="number of maximum uncertainty digits (default: 2)",
)
parser.add_argument(
    "-U",
    "--with-unc",
    dest="unc",
    action="store_true",
    help="consider uncertainties in constants",
)
parser.add_argument(
    "-C",
    "--no-corr",
    dest="corr",
    action="store_false",
    help="ignore correlations between constants",
)
parser.add_argument(
    "fr", metavar="from", type=str, help="unit or quantity to convert from"
)
parser.add_argument("to", type=str, nargs="?", help="unit to convert to")
try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    raise

ureg = UnitRegistry()
ureg.auto_reduce_dimensions = True
ureg.autoconvert_offset_to_baseunit = True
ureg.enable_contexts("Gau", "ESU", "sp", "energy", "boltzmann")
ureg.default_system = args.system


def _set(key: str, value) -> None:
    obj = ureg._units[key].converter
    object.__setattr__(obj, "scale", value)


if args.unc:
    try:
        import uncertainties
    except ImportError:
        raise Exception(
            "Failed to import uncertainties library!\n Please install uncertainties package"
        )

    # Measured constants subject to correlation
    #  R_i: Rydberg constant
    #  g_e: Electron g factor
    #  m_u: Atomic mass constant
    #  m_e: Electron mass
    #  m_p: Proton mass
    #  m_n: Neutron mass
    #  x_Cu: Copper x unit
    #  x_Mo: Molybdenum x unit
    #  A_s: Angstrom star
    R_i = (ureg._units["R_inf"].converter.scale, 0.0000000000012e7)
    g_e = (ureg._units["g_e"].converter.scale, 0.00000000000036)
    m_u = (ureg._units["m_u"].converter.scale, 0.00000000052e-27)
    m_e = (ureg._units["m_e"].converter.scale, 0.0000000028e-31)
    m_p = (ureg._units["m_p"].converter.scale, 0.00000000052e-27)
    m_n = (ureg._units["m_n"].converter.scale, 0.00000000085e-27)
    x_Cu = (ureg._units["x_unit_Cu"].converter.scale, 0.00000028e-13)
    x_Mo = (ureg._units["x_unit_Mo"].converter.scale, 0.00000053e-13)
    A_s = (ureg._units["angstrom_star"].converter.scale, 0.00000090e-10)
    if args.corr:
        # fmt: off
        # Correlation matrix between measured constants (to be completed below)
        #         R_i       g_e      m_u      m_e      m_p      m_n     x_Cu     x_Mo      A_s
        corr = [
            [ 1.00000, -0.00122, 0.00438, 0.00225, 0.00455, 0.00277, 0.00000, 0.00000, 0.00000],  # R_i
            [-0.00122,  1.00000, 0.97398, 0.97555, 0.97404, 0.59702, 0.00000, 0.00000, 0.00000],  # g_e
            [ 0.00438,  0.97398, 1.00000, 0.99839, 0.99965, 0.61279, 0.00000, 0.00000, 0.00000],  # m_u
            [ 0.00225,  0.97555, 0.99839, 1.00000, 0.99845, 0.61199, 0.00000, 0.00000, 0.00000],  # m_e
            [ 0.00455,  0.97404, 0.99965, 0.99845, 1.00000, 0.61281, 0.00000, 0.00000, 0.00000],  # m_p
            [ 0.00277,  0.59702, 0.61279, 0.61199, 0.61281, 1.00000,-0.00098,-0.00108,-0.00063],  # m_n
            [ 0.00000,  0.00000, 0.00000, 0.00000, 0.00000,-0.00098, 1.00000, 0.00067, 0.00039],  # x_Cu
            [ 0.00000,  0.00000, 0.00000, 0.00000, 0.00000,-0.00108, 0.00067, 1.00000, 0.00100],  # x_Mo
            [ 0.00000,  0.00000, 0.00000, 0.00000, 0.00000,-0.00063, 0.00039, 0.00100, 1.00000],  # A_s
        ]
        # fmt: on
        try:
            (
                R_i,
                g_e,
                m_u,
                m_e,
                m_p,
                m_n,
                x_Cu,
                x_Mo,
                A_s,
            ) = uncertainties.correlated_values_norm(
                [R_i, g_e, m_u, m_e, m_p, m_n, x_Cu, x_Mo, A_s], corr
            )
        except AttributeError:
            raise Exception(
                "Correlation cannot be calculated!\n  Please install numpy package"
            )
    else:
        R_i = uncertainties.ufloat(*R_i)
        g_e = uncertainties.ufloat(*g_e)
        m_u = uncertainties.ufloat(*m_u)
        m_e = uncertainties.ufloat(*m_e)
        m_p = uncertainties.ufloat(*m_p)
        m_n = uncertainties.ufloat(*m_n)
        x_Cu = uncertainties.ufloat(*x_Cu)
        x_Mo = uncertainties.ufloat(*x_Mo)
        A_s = uncertainties.ufloat(*A_s)

    _set("R_inf", R_i)
    _set("g_e", g_e)
    _set("m_u", m_u)
    _set("m_e", m_e)
    _set("m_p", m_p)
    _set("m_n", m_n)
    _set("x_unit_Cu", x_Cu)
    _set("x_unit_Mo", x_Mo)
    _set("angstrom_star", A_s)

    # Measured constants with zero correlation
    _set(
        "gravitational_constant",
        uncertainties.ufloat(
            ureg._units["gravitational_constant"].converter.scale, 0.00015e-11
        ),
    )

    ureg._root_units_cache = {}
    ureg._build_cache()


def convert(u_from, u_to=None, unc=None, factor=None) -> None:
    prec_unc = 0
    q = ureg.Quantity(u_from)
    fmt = f".{args.prec}g"
    if unc:
        q = q.plus_minus(unc)
    if u_to:
        nq = q.to(u_to)
    else:
        nq = q.to_base_units()
    if factor:
        q *= ureg.Quantity(factor)
        nq *= ureg.Quantity(factor).to_base_units()
    if args.unc:
        prec_unc = use_unc(nq.magnitude, fmt, args.prec_unc)
    if prec_unc > 0:
        fmt = f".{prec_unc}uS"
    else:
        with contextlib.suppress(Exception):
            nq = nq.magnitude.n * nq.units

    fmt = "{:" + fmt + "} {:~P}"
    print(("{:} = " + fmt).format(q, nq.magnitude, nq.units))


def use_unc(num, fmt, prec_unc):
    unc = 0
    with contextlib.suppress(Exception):
        if isinstance(num, uncertainties.UFloat):
            full = ("{:" + fmt + "}").format(num)
            unc = re.search(r"\+/-[0.]*([\d.]*)", full).group(1)
            unc = len(unc.replace(".", ""))

    return max(0, min(prec_unc, unc))


def main() -> None:
    convert(args.fr, args.to)


if __name__ == "__main__":
    main()
