.. _convert:

Command-line script
===================

The script `pint-convert` allows a quick conversion to a target system or
between arbitrary compatible units.

By default, `pint-convert` converts to SI units::

    $ pint-convert 225lb
    225 pound = 102.05828325 kg

use the `--sys` argument to change it::

    $ pint-convert --sys US 102kg
    102 kilogram = 224.871507429 lb

or specify directly the target units::

    $ pint-convert 102kg lb
    102 kilogram = 224.871507429 lb

The input quantity can contain expressions::

    $ pint-convert 7ft+2in
    7.166666666666667 foot = 2.1844 m

in some cases parentheses and quotes may be needed::

    $ pint-convert "225lb/(7ft+2in)"
    31.3953488372093 pound / foot = 46.7214261353 kg/m

If a number is omitted, 1 is assumed::

    $ pint-convert km mi
    1 kilometer = 0.621371192237 mi

The default precision is 12 significant figures, it can be changed with `-p`,
but note that the accuracy may be affected by floating-point errors::

    $ pint-convert -p 3 mi
    1 mile = 1.61e+03 m

    $ pint-convert -p 30 ly km
    1 light_year = 9460730472580.80078125 km

Some contexts are automatically enabled, allowing conversion between not fully
compatible units::

    $ pint-convert 540nm
    540 nanometer = 5.4e-07 m

    $ pint-convert kcal/mol
    $ 1.0 kilocalorie / mole = 4184 kg·m²/mol/s²

    $ pint-convert 540nm kcal/mol
    540 nanometer = 52.9471025594 kcal/mol

With the `uncertainties` package, the experimental uncertainty in the physical
constants is considered, and the result is given in compact notation, with the
uncertainty in the last figures in parentheses::

    $ pint-convert Eh eV
    1 hartree = 27.21138624599(5) eV

The precision is limited by both the maximum number of significant digits (`-p`)
and the maximum number of uncertainty digits (`-u`, 2 by default)::

    $ pint-convert -p 20 Eh eV
    1 hartree = 27.211386245988(52) eV

    $ pint-convert -p 20 -u 4 Eh eV
    1 hartree = 27.21138624598847(5207) eV

The uncertainty can be disabled with `-U`)::

    $ pint-convert -p 20 -U Eh eV
    1 hartree = 27.211386245988471444 eV

Correlations between experimental constants are also known, and taken into
account. Use `-C` to disable it::

    $ pint-convert --sys atomic m_p
    1 proton_mass = 1836.15267344(11) m_e

    $ pint-convert --sys atomic -C m_p
    1 proton_mass = 1836.15267344(79) m_e

Again, note that results may differ slightly, usually in the last figure, from
more authoritative sources, mainly due to floating-point errors.
