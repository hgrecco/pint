.. _faq:

.. currentmodule:: pint

Frequently asked questions
==========================


Why the name *Pint*?
--------------------

Pint is a unit and sounds like Python in the first syllable. Most important, it is a good unit for beer.


How can I avoid magnitude floating point issues?
-------------------------------------------------------------

By default, ``UnitRegistry`` uses ``float`` for magnitudes, which is subject
to the usual floating-point rounding behavior:

.. doctest::

   >>> import pint
   >>> ureg = pint.UnitRegistry()
   >>> ureg("1 gallon").to("l")
   Quantity(3.785411783999999, "liter")

If you need exact arithmetic instead, pass ``non_int_type`` when creating
the registry:

.. doctest::

   >>> import decimal
   >>> ureg = pint.UnitRegistry(non_int_type=decimal.Decimal)
   >>> ureg("1 gallon").to("l")
   Quantity(Decimal('3.785411784000000000000000000'), "liter")

or, for exact fractions:

.. doctest::

   >>> import fractions
   >>> ureg = pint.UnitRegistry(non_int_type=fractions.Fraction)

This avoids floating-point noise entirely (no rounding to begin with), at
the cost of some performance compared to plain ``float``.


How can I avoid exponent floating point issues?
-------------------------------------------------------------------------------------------------

Combining quantities raised to fractional powers (e.g. multiplying and
dividing several quantities each raised to a different fractional exponent)
can leave a tiny floating-point residual on an exponent that should be an
exact integer or zero, instead of the number of a whole quantity you might
expect, e.g. ``meter ** 0.9999999999999998`` instead of exactly ``meter ** 1``.

Converting the quantity with :py:meth:`Quantity.to`, :py:meth:`Quantity.to_base_units`,
or :py:meth:`Quantity.to_reduced_units` recomputes the units against the
requested target and resolves this: the result has exactly the units you
asked for, with no leftover noise.

.. note::

   Formatters may round the displayed exponent, hiding the issue rather than
   fixing it: ``meter ** 0.9999999999999998`` still *prints* as
   ``meter ** 1``, even though the stored exponent is not exactly ``1``.

Using Fraction as the exponent avoids this noise, however if this quantity is
then multiplied by a quantity which includes a float exponent, the same
noise reappears (see below). Ten multiplications by ``meter ** 0.1`` don't
quite add up to exactly ``meter ** 1``; ten multiplications by
``meter ** Fraction(1, 10)`` do:

.. doctest::

   >>> u = ureg.Unit("meter") ** 0.1
   >>> for _ in range(9):
   ...     u = u * ureg.Unit("meter") ** 0.1
   >>> u == ureg.Unit("meter")
   False

   >>> import fractions
   >>> ureg2 = pint.UnitRegistry(non_int_type=fractions.Fraction)
   >>> u2 = ureg2.Unit("meter") ** 0.1
   >>> for _ in range(9):
   ...     u2 = u2 * ureg2.Unit("meter") ** 0.1
   >>> u2 == ureg2.Unit("meter")
   False

   >>> u3 = ureg2.Unit("meter") ** fractions.Fraction(1, 10)
   >>> for _ in range(9):
   ...     u3 = u3 * ureg2.Unit("meter") ** fractions.Fraction(1, 10)
   >>> u3 == ureg2.Unit("meter")
   True

This comes at a performance cost (roughly 25% slower than plain ``float``
for typical arithmetic), and it only stays exact as long as every exponent
is exact: combining an exact ``Fraction`` exponent with an ordinary
``float`` exponent on the same unit produces a ``float`` result, the same
noise-prone type this is meant to avoid:

.. doctest::

   >>> Q_ = ureg2.Quantity
   >>> a = Q_(1, ureg2.m ** fractions.Fraction(8, 10))
   >>> b = Q_(1, ureg2.m ** 0.5)
   >>> c = a * b
   >>> c.units._units["meter"]
   1.3
   >>> type(c.units._units["meter"])
   <class 'float'>


You mention other similar Python libraries. Can you point me to those?
----------------------------------------------------------------------

`natu <https://kdavies4.github.io/natu/>`_

`Buckingham <https://github.com/mdipierro/buckingham>`_

`Magnitude <https://github.com/juanre/magnitude>`_

`SciMath <https://github.com/enthought/scimath>`_

`Python-quantities <https://github.com/python-quantities/python-quantities>`_

`Unum <https://bitbucket.org/kiv/unum>`_

`Units <https://bitbucket.org/adonohue/units/>`_

`udunitspy <https://github.com/blazetopher/udunitspy>`_

`SymPy <https://docs.sympy.org/latest/modules/physics/units/index.html>`_

`cf units <https://github.com/SciTools/cf_units>`_

`astropy units <https://github.com/astropy/astropy>`_

`yt <https://github.com/yt-project/yt>`_

`measurement <https://github.com/coddingtonbear/python-measurement>`_

If you're aware of another one, please contribute a patch to the docs.
