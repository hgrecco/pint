.. _serialization:


Serialization
=============

In order to dump a **Quantity** to disk, store it in a database or
transmit it over the wire you need to be able to serialize and then
deserialize the object.

The easiest way to do this is by converting the quantity to a string:

.. testsetup:: *

   import pint

.. doctest::

   >>> import pint
   >>> ureg = pint.UnitRegistry()
   >>> duration = 24.2 * ureg.years
   >>> duration
   <Quantity(24.2, 'year')>
   >>> serialized = str(duration)
   >>> print(serialized)
   24.2 year

Remember that you can easily control the number of digits in the representation
as shown in :ref:`sec-string-formatting`.

You dump/store/transmit the content of serialized ('24.2 year'). When you want
to recover it in another process/machine, you just:

.. doctest::

   >>> import pint
   >>> ureg = pint.UnitRegistry()
   >>> duration = ureg('24.2 year')
   >>> print(duration)
   24.2 year

Notice that the serialized quantity is likely to be parsed in **another** registry
as shown in this example. Pint Quantities do not exist on their own but they are
always related to a **UnitRegistry**. Everything will work as expected if both registries,
are compatible (e.g. they were created using the same definition file). However, things
could go wrong if the registries are incompatible. For example, **year** could not be
defined in the target registry. Or what is even worse, it could be defined in a different
way. Always have to keep in mind that the interpretation and conversion of Quantities are
UnitRegistry dependent.

In certain cases, you want a binary representation of the data. Python's standard algorithm
for serialization is called Pickle_. Pint quantities implement the magic `__reduce__`
method and therefore can be *Pickled* and *Unpickled*. However, you have to bear in mind, that
the **DEFAULT_REGISTRY** is used for unpickling and this might be different from the one
that was used during pickling. If you want to have control over the deserialization, the
best way is to create a tuple with the magnitude and the units:

.. doctest::

    >>> to_serialize = duration.to_tuple()
    >>> print(to_serialize)
    (24.2, (('year', 1.0),))

And then you can just pickle that:

    >>> import pickle
    >>> serialized = pickle.dumps(to_serialize, -1)

To unpickle, just

    >>> loaded = pickle.loads(serialized)
    >>> ureg.Quantity.from_tuple(loaded)
    <Quantity(24.2, 'year')>

(To pickle to and from a file just use the dump and load method as described in _Pickle)

You can use the same mechanism with any serialization protocol, not only with binary ones.
(In fact, version 0 of the Pickle protocol is ASCII). Other common serialization protocols/packages
are json_, yaml_, shelve_, hdf5_ (or via PyTables_) and dill_.
Notice that not all of these packages will serialize properly the magnitude (which can be any
numerical type such as `numpy.ndarray`).

Using the serialize_ package you can load and read from multiple formats:

    >>> from serialize import dump, load, register_class
    >>> register_class(ureg.Quantity, ureg.Quantity.to_tuple, ureg.Quantity.from_tuple)
    >>> dump(duration, 'output.yaml')
    >>> r = load('output.yaml')

(Check out the serialize_ docs for more information)


.. _serialize: https://github.com/hgrecco/serialize
.. _Pickle: http://docs.python.org/3/library/pickle.html
.. _json: http://docs.python.org/3/library/json.html
.. _yaml: http://pyyaml.org/
.. _shelve: http://docs.python.org/3.4/library/shelve.html
.. _hdf5: http://www.h5py.org/
.. _PyTables: http://www.pytables.org
.. _dill: https://pypi.python.org/pypi/dill



