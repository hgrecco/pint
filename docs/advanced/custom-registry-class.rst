.. _custom_registry_class:

Custom registry class
=====================

Pay as you go
-------------

Pint registry functionality is divided into facets. The default
UnitRegistry inherits from all of them, providing a full fledged
and feature rich registry. However, in certain cases you might want
to have a simpler and light registry. Just pick what you need
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
    >>> class MyRegistry(pint.facets.NonMultiplicativeRegistry, pint.facets.PlainRegistry):
    ...     pass


Subclassing
-----------

If you want to add the default registry class some specific functionality,
you can subclass it:

.. doctest::

    >>> import pint
    >>> class MyRegistry(pint.UnitRegistry):
    ...
    ...     def my_specific_function(self):
    ...         """Do something
    ...         """


If you want to create your own Quantity class, you must tell
your registry about it:

.. doctest::

    >>> import pint
    >>> class MyQuantity:
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
    >>>
    >>> class MyRegistry(pint.UnitRegistry):
    ...
    ...     _quantity_class = MyQuantity
    ...
    ...     # The same you can be done with
    ...     # _unit_class
    ...     # _measurement_class


While these examples demonstrate how to add functionality to the default
registry class, you can actually subclass just the PlainRegistry or any
combination of facets.
