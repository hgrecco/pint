.. _angular_frequency:


Angular Frequency
=================

`Hertz` is a unit for frequency, that is often also used for angular frequency. For example, a shaft spinning at `60 revolutions per minute` will often be said to spin at `1 Hz`, rather than `1 revolution per second`.

By default, pint treats angle quantities as `dimensionless`, so allows conversions between frequencies and angular frequencies. The base unit for angle is the `radian`. This leads to some unintuitive behaviour, as pint will convert angular frequencies into frequencies by converting angles into `radians`, rather than `revolutions`. This leads to converted values `2 * pi` larger than expected:

.. code-block:: python

    >> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
    >>> angular_frequency = ureg('60rpm')
    >>> angular_frequency.to('Hz')
    <Quantity(6.28318531, 'hertz')>

pint follows the conventions of SI. The SI BIPM Brochure (Bureau International des Poids et Mesures) states:

.. note::

    The SI unit of frequency is hertz, the SI unit of angular velocity and angular frequency is
    radian per second, and the SI unit of activity is becquerel, implying counts per second.
    Although it is formally correct to write all three of these units as the reciprocal second, the
    use of the different names emphasizes the different nature of the quantities concerned. It is
    especially important to carefully distinguish frequencies from angular frequencies, because
    by definition their numerical values differ by a factor1 of 2π. Ignoring this fact may cause
    an error of 2π. Note that in some countries, frequency values are conventionally expressed
    using “cycle/s” or “cps” instead of the SI unit Hz, although “cycle” and “cps” are not units
    in the SI. Note also that it is common, although not recommended, to use the term
    frequency for quantities expressed in rad/s. Because of this, it is recommended that
    quantities called “frequency”, “angular frequency”, and “angular velocity” always be given
    explicit units of Hz or rad/s and not s−1
