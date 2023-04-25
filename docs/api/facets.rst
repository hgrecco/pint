API facets reference
====================

Registry functionality is divided into facet. Each provides classes and functions
specific to a particular purpose. They expose at least a Registry, and in certain
cases also a Quantity, Unit and other objects.

The default UnitRegistry inherits from all of them.


.. automodule:: pint.facets.plain
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System

.. automodule:: pint.facets.nonmultiplicative
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System

.. automodule:: pint.facets.formatting
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System

.. automodule:: pint.facets.numpy
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System

.. automodule:: pint.facets.dask
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System

.. automodule:: pint.facets.measurement
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System

.. automodule:: pint.facets.group
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System

.. automodule:: pint.facets.system
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System

.. automodule:: pint.facets.context
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System
