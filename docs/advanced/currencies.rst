.. _currencies:

Using Pint for currency conversions
===================================
Currency conversion tends to be substantially more complex than physical units.
The exact exchange rate between two currencies:

- changes every minute,
- changes depending on the place,
- changes depending on who you are and who changes the money,
- may be not reversible, e.g. EUR->USD may not be the same as 1/(USD->EUR),
- different rates may apply to different amounts of money, e.g. in a BUY/SELL ledger,
- frequently involves fees, whose calculation can be more or less sophisticated.
  For example, a typical credit card contract may state that the bank will charge you a
  fee on all purchases in foreign currency of 1 USD or 2%, whichever is higher, for all
  amounts less than 1000 USD, and then 1.5% for anything in excess.

You may implement currencies in two ways, both of which require you to be familiar
with :ref:`contexts`.

Simplified model
----------------

This model implies a few strong assumptions:

- There are no conversion fees
- All exchange rates are reversible
- Any amount of money can be exchanged at the same rate
- All exchanges can happen at the same time, between the same actors.

In this simplified scenario, you can perform any round-trip across currencies
and always come back with the original money; e.g.
1 USD -> EUR -> JPY -> GBP -> USD will always give you 1 USD.

In reality, these assumptions almost never happen but can be a reasonable approximation,
for example in the case of large financial institutions, which can use interbank
exchange rates and have nearly-limitless liquidity and sub-second trading systems.

This can be implemented by putting all currencies on the same dimension, with a
default conversion rate of NaN, and then setting the rate within contexts::

    USD = [currency]
    EUR = nan USD
    JPY = nan USD
    GBP = nan USD

    @context FX
        EUR = 1.11254 USD
        GBP = 1.16956 EUR
    @end

Note how, in the example above:

- USD is our *base currency*. It is arbitrary, only matters for the purpose
  of invoking ``to_base_units()``, and can be changed with :ref:`systems`.
- We did not set a value for JPY - maybe because the trader has no availability, or
  because the data source was for some reason missing up-to-date data.
  Any conversion involving JPY will return NaN.
- We redefined GBP to be a function of EUR instead of USD. This is fine as long as there
  is a path between two currencies.

Full model
----------

If any of the assumptions of the simplified model fails, one can resort to putting each
currency on its own dimension, and then implement transformations::

    EUR = [currency_EUR]
    GBP = [currency_GBP]

    @context FX
        GBP -> EUR: value * 1.11108 EUR/GBP
        EUR -> GBP: value * 0.81227 GBP/EUR
    @end

.. code-block:: python

    >>> q = ureg.Quantity("1 EUR")
    >>> with ureg.context("FX"):
    ... q = q.to("GBP").to("EUR")
    >>> q
    0.9024969516 EUR

More sophisticated formulas, e.g. dealing with flat fees and thresholds, can be
implemented with arbitrary python code by programmatically defining a context (see
:ref:`contexts`).

Currency Symbols
----------------

Many common currency symbols are not supported by the pint parser. A preprocessor can be used as a workaround:

.. doctest::

   >>> import pint
   >>> ureg = pint.UnitRegistry(preprocessors = [lambda s: s.replace("€", "EUR")])
   >>> ureg.define("euro = [currency] = € = EUR")
   >>> print(ureg.Quantity("1 €"))
   1 euro

Example using Currency Converter
-------------------------------

The following example demonstrates how to use `currency_converter` with Pint to perform currency conversions with historical rates.

.. doctest::

   >>> import itertools
   >>> import pint
   >>> import currency_converter
   >>> from datetime import date
   >>>
   >>> # create a currency converter instance to load all of the data
   >>> cc = currency_converter.CurrencyConverter(fallback_on_missing_rate=True)
   >>>
   >>> # load custom units and instantiate Quantity base class that is used everywhere
   >>> ureg = pint.UnitRegistry()
   >>> for c in cc.currencies:
   ...     ureg.define(f"{c} = [currency_{c}]")  # i.e. USD = [currency_USD]
   >>>
   >>> # add programmatic context for currency conversion
   >>> currency_context = pint.Context("FX", defaults={"date": None})
   >>> for a, b in itertools.combinations(list(cc.currencies), 2):
   ...     def a2b(_ureg, x, date=None, a=a, b=b):
   ...         return cc.convert(x.magnitude, a, b, date=date) * _ureg(b)
   ...     def b2a(_ureg, x, date=None, a=a, b=b):
   ...         return cc.convert(x.magnitude, b, a, date=date) * _ureg(a)
   ...     currency_context.add_transformation(f"[currency_{a}]", f"[currency_{b}]", a2b)
   ...     currency_context.add_transformation(f"[currency_{b}]", f"[currency_{a}]", b2a)
   >>> ureg.add_context(currency_context)
   >>>
   >>> # Example usage
   >>> q = ureg.Quantity("1 EUR")
   >>> with ureg.context("FX", date=date(2010, 11, 21)):
   ...     q = q.to("USD")
   >>> print(q)
   1.3656 USD

Note that the `currency_converter` package is not a dependency of Pint, so you will need to install it separately, for example with `pip install currencyconverter`.

.. _`currency_converter`: https://alexprengere.github.io/currencyconverter/
