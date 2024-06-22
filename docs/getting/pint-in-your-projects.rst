.. _pint_in_your_projects:

Using Pint in your projects
===========================

Having a shared registry
------------------------

If you use Pint in multiple modules within your Python package, you normally
want to avoid creating multiple instances of the unit registry.
The best way to do this is by instantiating the registry in a single place. For
example, you can add the following code to your package ``__init__.py``

.. doctest::

   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry()
   >>> Q_ = ureg.Quantity


Then in ``yourmodule.py`` the code would be

.. code-block:: python

   from . import ureg, Q_

   length = 10 * ureg.meter
   my_speed = Q_(20, 'm/s')

If you are pickling and unpickling Quantities within your project, you should
also define the registry as the application registry

.. code-block:: python

   from pint import UnitRegistry, set_application_registry
   ureg = UnitRegistry()
   set_application_registry(ureg)


.. warning:: There are no global units in Pint. All units belong to a registry and
    you can have multiple registries instantiated at the same time. However, you
    are not supposed to operate between quantities that belong to different registries.
    Never do things like this:

.. doctest::

   >>> q1 = 10 * UnitRegistry().meter
   >>> q2 = 10 * UnitRegistry().meter
   >>> q1 + q2
   Traceback (most recent call last):
   ...
   ValueError: Cannot operate with Quantity and Quantity of different registries.
   >>> id(q1._REGISTRY) == id(q2._REGISTRY)
   False


Keeping up to date with Pint development
----------------------------------------

While we work hard to avoid breaking code using Pint, sometimes it
happens. To help you track how Pint is evolving it is recommended
that you run a daily or weekly job against pint master branch.

For example, this is how xarray_ is doing it:

If a new version of Pint breaks your code, please open an issue_ to
let us know.

.. _xarray: https://github.com/pydata/xarray/blob/main/.github/workflows/upstream-dev-ci.yaml
.. _issue: https://github.com/hgrecco/pint/issues
