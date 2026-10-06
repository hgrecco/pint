.. _typing:

Typing and IDE Autocompletion
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

When ``ureg.wraps`` receives a single return unit (a string or a Unit), the
decorated function's return type preserves the registry's Quantity type.
For example, array quantities returned by a ``UnitRegistry`` wrapper can be indexed.
With ``ret=None`` or multiple return units, the return type is conservatively
annotated as ``object``; skipped conversions and mixed return values need not
produce a Quantity. These annotations do not change the runtime behavior.
