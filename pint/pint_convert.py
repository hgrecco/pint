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


def _set(key: str, value):
    obj = ureg._units[key].converter
    object.__setattr__(obj, "scale", value)


if args.unc:
    try:
        import uncertainties
    except ImportError:
        raise Exception(
            "Failed to import uncertainties library!\n Please install uncertainies"
        )

    # Measured constants subject to correlation
    #  R_i: Rydberg constant
    #  g_e: Electron g factor
    #  m_u: Atomic mass constant
    #  m_e: Electron mass
    #  m_p: Proton mass
    #  m_n: Neutron mass
    R_i = (ureg._units["R_inf"].converter.scale, 0.0000000000021e7)
    g_e = (ureg._units["g_e"].converter.scale, 0.00000000000035)
    m_u = (ureg._units["m_u"].converter.scale, 0.00000000050e-27)
    m_e = (ureg._units["m_e"].converter.scale, 0.00000000028e-30)
    m_p = (ureg._units["m_p"].converter.scale, 0.00000000051e-27)
    m_n = (ureg._units["m_n"].converter.scale, 0.00000000095e-27)
    if args.corr:
        # Correlation matrix between measured constants (to be completed below)
        #          R_i       g_e       m_u       m_e       m_p       m_n
        corr = [
            [1.0, -0.00206, 0.00369, 0.00436, 0.00194, 0.00233],  # R_i
            [-0.00206, 1.0, 0.99029, 0.99490, 0.97560, 0.52445],  # g_e
            [0.00369, 0.99029, 1.0, 0.99536, 0.98516, 0.52959],  # m_u
            [0.00436, 0.99490, 0.99536, 1.0, 0.98058, 0.52714],  # m_e
            [0.00194, 0.97560, 0.98516, 0.98058, 1.0, 0.51521],  # m_p
            [0.00233, 0.52445, 0.52959, 0.52714, 0.51521, 1.0],
        ]  # m_n
        try:
            (R_i, g_e, m_u, m_e, m_p, m_n) = uncertainties.correlated_values_norm(
                [R_i, g_e, m_u, m_e, m_p, m_n], corr
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

    _set("R_inf", R_i)
    _set("g_e", g_e)
    _set("m_u", m_u)
    _set("m_e", m_e)
    _set("m_p", m_p)
    _set("m_n", m_n)

    # Measured constants with zero correlation
    _set(
        "gravitational_constant",
        uncertainties.ufloat(
            ureg._units["gravitational_constant"].converter.scale, 0.00015e-11
        ),
    )

    _set(
        "d_220",
        uncertainties.ufloat(ureg._units["d_220"].converter.scale, 0.000000032e-10),
    )

    _set(
        "K_alpha_Cu_d_220",
        uncertainties.ufloat(
            ureg._units["K_alpha_Cu_d_220"].converter.scale, 0.00000022
        ),
    )

    _set(
        "K_alpha_Mo_d_220",
        uncertainties.ufloat(
            ureg._units["K_alpha_Mo_d_220"].converter.scale, 0.00000019
        ),
    )

    _set(
        "K_alpha_W_d_220",
        uncertainties.ufloat(
            ureg._units["K_alpha_W_d_220"].converter.scale, 0.000000098
        ),
    )

    ureg._root_units_cache = {}
    ureg._build_cache()


def convert(u_from, u_to=None, unc=None, factor=None):
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


def main():
    convert(args.fr, args.to)


if __name__ == "__main__":
    main()
