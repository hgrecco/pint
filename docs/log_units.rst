.. _log_units:


Logarithmic Units
=================

.. warning::

    Support for logarithmic units in Pint is currently in Beta. Please take
    careful note of the information below, particularly around `compound log units`_
    to avoid calculation errors. Bug reports and pull requests are always
    welcome, please see :doc:`contributing` for more information on
    how you can help improve this feature (and Pint in general).

Pint supports some logarithmic units, including `dB`, `dBm`, `octave`, and `decade`
as well as some conversions between them and their base units where applicable.
These units behave much like those described in :ref:`nonmult`, so many of
the recommendations there apply here as well.

Mathematical operations with logarithmic units are often ambiguous.
For example, the sum of two powers with decibel units is a logarithmic quantity of the power squared, thus without obvious meaning and not decibel units.
Therefore the main Pint distribution raises an `OffsetUnitCalculusError` as a result of the sum of two quantities with decibel units,
as it does for all other ambiguous mathematical operations.

Valispace's fork of Pint makes some options.
We distiguish between *absolute logarithmic units* and *relative logarithmic units*.

Absolute logarithmic units are the logarithmic units with a constant reference, e.g. `dBW` corresponds to a power change in relation to 1 `Watt`.
We consider general logarithmic units like `dB` as general absolute logarithmic units.

Relative logarithmic units are the logarithmic units of gains and losses, thus a power change in relation to the previous power level.
In coherence with the default behaviour of subtraction between absolute logarithmic units,
relative logarithmic units are represented by `delta_` before the name of the correspondent absolute logarithmic unit, e.g. `delta_dBu` corresponds to a power change in relation to a power level in `dBu`.

Setting up the ``UnitRegistry()``
---------------------------------

Many of the examples below will fail without supplying the
``autoconvert_offset_to_baseunit=True`` flag. To use logarithmic units,
intialize your ``UnitRegistry()`` like so:

.. doctest::

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
    >>> Q_ = ureg.Quantity

If you can't pass that flag you will need to define all logarithmic units
:ref:`using the Quantity() constructor<Using the constructor>`, and you will
be restricted in the kinds of operations you can do without explicitly calling
`.to_base_units()` first.

The sum of decibel absolute units will raise an error by default.
However, you can set the registry flag `logarithmic_math` to `True` when starting the unit registry, like:

.. doctest::

    >>> ureg = UnitRegistry(autoconvert_offset_to_baseunit=True, logarithmic_math=True)

If you switch on this flag, it will convert additions of quantities with logarithmic units into logarithmic additions.

Defining log quantities
-----------------------

After you've set up your ``UnitRegistry()`` with the ``autoconvert...`` flag,
you can define simple logarithmic quantities like most others:

.. doctest::

    >>> 20.0 * ureg.dBm
    <Quantity(20.0, 'decibelmilliwatt')>
    >>> ureg('20.0 dBm')
    <Quantity(20.0, 'decibelmilliwatt')>
    >>> ureg('20 dB')
    <Quantity(20, 'decibel')>
    >>> ureg('20 delta_dB')
    <Quantity(20, 'delta_decibel')>


Converting to and from base units
---------------------------------

Get a sense of how logarithmic units are handled by using the `.to()` and
`.to_base_units()` methods:

.. doctest::

    >>> ureg('20 dBm').to('mW')
    <Quantity(100.0, 'milliwatt')>
    >>> ureg('20 dB').to_base_units()
    <Quantity(100.0, 'dimensionless')>

.. note::

    Notice in the above example how the `dB` unit is defined for
    power quantities (10*log(p/p0)) not field (amplitude) quantities
    (20*log(v/v0)). Take care that you're only using it to multiply power
    levels, and not e.g. Voltages.

Convert back from a base unit to a logarithmic unit using the `.to()` method:

.. doctest::

    >>> (100.0 * ureg('mW')).to('dBm')
    <Quantity(20.0, 'decibelmilliwatt')>
    >>> shift = Q_(4, '')
    >>> shift
    <Quantity(4, 'dimensionless')>
    >>> shift.to('octave')
    <Quantity(2.0, 'octave')>

Compound log units
------------------

.. warning::

    Support for compound logarithmic units is not comprehensive. The following
    examples work, but many others will not. Consider converting the logarithmic
    portion to base units before adding more units.

Pint sometimes works with mixtures of logarithmic and other units. Below is an
example of computing RMS noise from a noise density and a bandwidth:

.. doctest::

    >>> noise_density = -161.0 * ureg.dBm / ureg.Hz
    >>> bandwidth = 10.0 * ureg.kHz
    >>> noise_power = noise_density * bandwidth
    >>> noise_power.to('dBm')
    <Quantity(-121.0, 'decibelmilliwatt')>
    >>> noise_power.to('mW')
    <Quantity(7.94328235e-13, 'milliwatt')>

There are still issues with parsing compound units, so for now the following
will not work:

.. doctest::

    >>> -161.0 * ureg('dBm/Hz') == (-161.0 * ureg.dBm / ureg.Hz)
    False

But this will:

.. doctest::

    >>> ureg('-161.0 dBm/Hz') == (-161.0 * ureg.dBm / ureg.Hz)
    True
    >>> Q_(-161.0, 'dBm') / ureg.Hz == (-161.0 * ureg.dBm / ureg.Hz)
    True

To begin using this feature while avoiding problems, define logarithmic units
as single-unit quantities and convert them to their base units as quickly as
possible.
