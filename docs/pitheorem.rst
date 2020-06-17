.. _pitheorem:

Buckingham Pi Theorem
=====================

`Buckingham π theorem`_ states that an equation involving *n* number of
physical variables which are expressible in terms of *k* independent fundamental
physical quantities can be expressed in terms of *p = n - k* dimensionless
parameters.

.. testsetup:: *

   from pint import UnitRegistry
   ureg = UnitRegistry()
   Q_ = ureg.Quantity

To start with a very simple case, consider that you want to find a dimensionless
quantity involving the magnitudes `V`, `T` and `L` with dimensions `[length]/[time]`,
`[time]` and `[length]` respectively.

.. doctest::

    >>> from pint import pi_theorem
    >>> pi_theorem({'V': '[length]/[time]', 'T': '[time]', 'L': '[length]'})
    [{'V': 1.0, 'T': 1.0, 'L': -1.0}]

The result indicates that a dimensionless quantity can be obtained by
multiplying `V` by `T` and the inverse of `L`.

Which can be pretty printed using the `Pint` formatter:

.. doctest::

    >>> from pint import formatter
    >>> result = pi_theorem({'V': '[length]/[time]', 'T': '[time]', 'L': '[length]'})
    >>> print(formatter(result[0].items()))
    T * V / L

You can also apply the Buckingham π theorem associated to a Registry. In this case,
you can use derived dimensions such as speed:

.. doctest::

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
    >>> ureg.pi_theorem({'V': '[speed]', 'T': '[time]', 'L': '[length]'})
    [{'V': 1.0, 'T': 1.0, 'L': -1.0}]

or unit names:

.. doctest::

    >>> ureg.pi_theorem({'V': 'meter/second', 'T': 'second', 'L': 'meter'})
    [{'V': 1.0, 'T': 1.0, 'L': -1.0}]

or quantities:

    >>> Q_ = ureg.Quantity
    >>> ureg.pi_theorem({'V': Q_(1, 'meter/second'),
    ...                  'T': Q_(1, 'second'),
    ...                  'L': Q_(1, 'meter')})
    [{'V': 1.0, 'T': 1.0, 'L': -1.0}]


Application to the pendulum
---------------------------

There are 3 fundamental physical units in this equation: time, mass, and length, and 4 dimensional variables, T (oscillation period), M (mass), L (the length of the string), and g (earth gravity). Thus we need only 4 − 3 = 1 dimensionless parameter.

.. doctest::

    >>> ureg.pi_theorem({'T': '[time]',
    ...                  'M': '[mass]',
    ...                  'L': '[length]',
    ...                  'g': '[acceleration]'})
    [{'T': 2.0, 'L': -1.0, 'g': 1.0}]

which means that the dimensionless quantity is:

.. math::

   \Pi = \frac{g T^2}{L}

and therefore:

.. math::

    T = constant \sqrt{\frac{L}{g}}

(In case you wonder, the constant is equal to 2 π, but this is outside the scope of this help)


Pressure loss in a pipe
-----------------------

What is the pressure loss `p` in a pipe with length `L` and diameter `D` for a fluid with density `d`, and viscosity `m` travelling with speed `v`? As pressure, mass, volume, viscosity and speed are defined as derived dimensions in the registry, we only need to explicitly write the density dimensions.

.. doctest::

    >>> ureg.pi_theorem({'p': '[pressure]',
    ...                  'L': '[length]',
    ...                  'D': '[length]',
    ...                  'd': '[mass]/[volume]',
    ...                  'm': '[viscosity]',
    ...                  'v': '[speed]'
    ...                  })                             # doctest: +SKIP
    [{'p': 1.0, 'm': -2.0, 'd': 1.0, 'L': 2.0}, {'v': 1.0, 'm': -1.0, 'd': 1.0, 'L': 1.0}, {'L': -1.0, 'D': 1.0}]

The second dimensionless quantity is the `Reynolds Number`_

.. _`Buckingham π theorem`: http://en.wikipedia.org/wiki/Buckingham_%CF%80_theorem
.. _`Reynolds Number`: http://en.wikipedia.org/wiki/Reynolds_number
