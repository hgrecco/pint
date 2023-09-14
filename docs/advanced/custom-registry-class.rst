.. _custom_registry_class:

Custom registry class
=====================

Pay as you go
-------------

Pint registry functionality is divided into facets. The default
UnitRegistry inherits from all of them, providing a full fledged
and feature rich registry. However, in certain cases you might want
to have a simpler and lighter registry. Just pick what you need
and create your own.

- FormattingRegistry: adds the capability to format quantities and units into string.
- SystemRegistry: adds the capability to work with system of units.
- GroupRegistry: adds the capability to group units.
- MeasurementRegistry: adds the capability to handle measurements (quantities with uncertainties).
- NumpyRegistry: adds the capability to interoperate with NumPy.
- DaskRegistry: adds the capability to interoperate with Dask.
- ContextRegistry: the capability to contexts: predefined conversions
  between incompatible dimensions.
- NonMultiplicativeRegistry: adds the capability to handle nonmultiplicative units (offset, logarithmic).
- PlainRegistry: base implementation for registry, units and quantities.

The only required one is `PlainRegistry`, the rest are completely
optional.

For example:

.. doctest::

    >>> import pint
    >>> class MyRegistry(pint.facets.NonMultiplicativeRegistry):
    ...     pass


.. note::
   `NonMultiplicativeRegistry` is a subclass from `PlainRegistry`, and therefore
   it is not required to add it explicitly to `MyRegistry` bases.


You can add some specific functionality to your new registry.

.. doctest::

    >>> import pint
    >>> class MyRegistry(pint.UnitRegistry):
    ...
    ...     def my_specific_function(self):
    ...         """Do something
    ...         """



Custom Quantity and Unit class
------------------------------

You can also create your own Quantity and Unit class, you must derive
from Quantity (or Unit) and tell your registry about it.

For example, if you want to create a new `UnitRegistry` subclass you
need to  derive the Quantity and Unit classes from it.

.. doctest::

    >>> import pint
    >>> class MyQuantity(pint.UnitRegistry.Quantity):
    ...
    ...     # Notice that subclassing pint.Quantity
    ...     # is not necessary.
    ...     # Pint will inspect the Registry class and create
    ...     # a Quantity class that contains all the
    ...     # required parents.
    ...
    ...     def to_my_desired_format(self):
    ...         """Do something else
    ...         """
    ...
    >>> class MyUnit(pint.UnitRegistry.Unit):
    ...
    ...     # Notice that subclassing pint.Quantity
    ...     # is not necessary.
    ...     # Pint will inspect the Registry class and create
    ...     # a Quantity class that contains all the
    ...     # required parents.
    ...
    ...     def to_my_desired_format(self):
    ...         """Do something else
    ...         """


Then, you need to create a custom registry but deriving from `GenericUnitRegistry` so you
can specify the types of

.. doctest::

    >>> from typing_extensions import TypeAlias # Python 3.9
    >>> # from typing import TypeAlias # Python 3.10+
    >>> class MyRegistry(pint.registry.GenericUnitRegistry[MyQuantity, pint.Unit]):
    ...
    ...     Quantity: TypeAlias = MyQuantity
    ...     Unit: TypeAlias = MyUnit
    ...

While these examples demonstrate how to add functionality to the default
registry class, you can actually subclass just the `PlainRegistry`, and
`GenericPlainRegistry`.
