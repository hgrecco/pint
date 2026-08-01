.. _faq:

.. currentmodule:: pint

Frequently asked questions
==========================


Why the name *Pint*?
--------------------

Pint is a unit and sounds like Python in the first syllable. Most important, it is a good unit for beer.


How can I avoid floating-point precision issues altogether?
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


Why does my quantity have a unit with a strange exponent like ``meter ** 0.9999999999999998``?
-------------------------------------------------------------------------------------------------

Combining quantities raised to fractional powers (e.g. multiplying and
dividing several quantities each raised to a different fractional exponent)
can leave a tiny floating-point residual on an exponent that should be an
exact integer or zero, instead of the number of a whole quantity you might
expect.

Converting the quantity with :py:meth:`Quantity.to`, :py:meth:`Quantity.to_base_units`,
or :py:meth:`Quantity.to_reduced_units` recomputes the units against the
requested target and resolves this: the result has exactly the units you
asked for, with no leftover noise.


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
