
Base API
========

.. currentmodule:: pint


Most important classes
----------------------

.. autoclass:: UnitRegistry
    :members:
    :exclude-members: Quantity, Unit, Measurement, Group, Context, System

.. autoclass:: Quantity
    :members:

.. autoclass:: Unit
    :members:

.. autoclass:: Measurement
    :members:


Exceptions and warnings
-----------------------

.. autoexception:: PintError
    :members:

.. autoexception:: DefinitionSyntaxError
    :members:

.. autoexception:: LogarithmicUnitCalculusError
    :members:

.. autoexception:: DimensionalityError
    :members:

.. autoexception:: OffsetUnitCalculusError
    :members:

.. autoexception:: RedefinitionError
    :members:

.. autoexception:: UndefinedUnitError
    :members:

.. autoexception:: UnitStrippedWarning
    :members:


Sharing registry among packages
-------------------------------
,
.. autofunction:: get_application_registry
.. autofunction:: set_application_registry

Other functions
---------------

.. autofunction:: formatter
.. autofunction:: register_unit_format
.. autofunction:: pi_theorem
.. autoclass:: Context
