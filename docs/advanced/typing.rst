.. _typing:

Wrapping and checking functions
===============================

Type Annotations
----------------

Pint's Quantity class supports type annotations, which can be used to specify the type of
the magnitude (e.g., float, int, np.ndarray)


.. doctest::

    >>> import numpy as np
    >>> import pint
    >>> def my_scalar_func(x: pint.Quantity[float]) -> pint.Quantity[float]:
    ...     pass
    >>> def my_array_func(x: pint.Quantity[np.ndarray[(3, ), int]]) -> pint.Quantity[np.ndarray[(3, ), int]]:
    ...     pass
##