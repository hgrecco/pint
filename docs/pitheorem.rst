.. _pitheorem:

Buckingham Pi Theorem
=====================

'Buckingham π theorem'_ states that an equation involving *n* number of
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
you can use the unit names:

.. doctest::

    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
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
    [{'T': 2.0, 'g': 1.0, 'L': -1.0}]



.. _'Buckingham π theorem': http://en.wikipedia.org/wiki/Buckingham_%CF%80_theorem
