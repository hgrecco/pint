.. _log_units:


Logarithmic Units
=================

Pint supports some logarithmic units, including dB, dBm, octave, and decade
as well as conversions between them and their base units where applicable.
These units behave much like those described in :ref:`nonmult`, so many of
the recommendations there apply here as well.

.. note::

    If you're making heavy use of logarithmic units, you may find it helpful to
    pass ``autoconvert_offset_to_baseunit=True`` when initializing your ``UnitRegistry()``,
    this will allow you to use syntax like ``10.0 * ureg.dBm`` in lieu of the
    explicit ``Quantity()`` constructor. Many examples on this page assume
    you've passed this parameter, and will not work otherwise.

Defining log units
------------------

First, set up your ``UnitRegistry`` with the suggested flag.

If you do not wish to use the ``autoconvert_offset_to_baseunit`` flag, you
will need to define all logarithmic units using the ``Quanity()`` constructor:

.. doctest::

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
    >>> Q_ = ureg.Quantity
    >>> signal_power_dbm = 20 * ureg.dBm    # must pass flag for this to work
    Traceback (most recent call last):
        ...
    OffsetUnitCalculusError: Ambiguous operation with offset unit (decibellmilliwatt, ).
    >>> Q_(20, 'dBm')                       # define like this instead
    <Quantity(20, 'decibellmilliwatt')>

You will also be restricted in the kinds of operations you can do without
converting to base units first.

.. doctest::

    >>> Q_(10, 'dBm/Hz') * (100 * ureg.Hz)  # not feasible without flag
    Traceback (most recent call last):
        ...
    UndefinedUnitError: 'delta_decibellmilliwatt' is not defined in the unit registry

Passing the flag will allow you to use a more natural syntax for defining
logarithmic units:

.. doctest::

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
    >>> Q_ = ureg.Quantity
    >>> 20.0 * ureg.dBm
    <Quantity(20.0, 'decibellmilliwatt')>

Converting to and from base units
---------------------------------

    >>> signal_power_dbm = 20.0 * ureg.dBm
    >>> signal_power_dbm.to('mW')
    <Quantity(100.0, 'milliwatt')>
    >>> signal_power_mw = 100.0 * ureg.mW
    >>> signal_power_mw
    <Quantity(100.0, 'milliwatt')>
    >>> signal_power_mw.to('dBm')
    <Quantity(20.0, 'decibellmilliwatt')>

Compound log units
------------------

Pint also works with mixtures of logarithmic and other units.

.. doctest::

    >>> noise_density = -161.0 * ureg['dBm/Hz']
    >>> bandwidth = 10.0 * ureg.kHz
    >>> noise_power = noise_density * bandwidth
    >>> noise_power.to('dBm')
    <Quantity(-121.0, 'decibellmilliwatt')>
    >>> noise_power.to('mW')
    <Quantity(7.94328235e-13, 'milliwatt')>

Multiplication, division and exponentiation of quantities with
offset units is problematic just like addition. Pint (since version 0.6)
will by default raise an error when a quantity with offset unit is used in
these operations. Due to this quantities with offset units cannot be created
like other quantities by multiplication of magnitude and unit but have
to be explicitly created:
