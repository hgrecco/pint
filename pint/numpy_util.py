from .util import iterable, sized
from .compat import np


def is_quantity_with_scalar_magnitude(obj):
    """Test for Quantity with scalar magnitude.

    Parameters
    ----------
    obj : object


    Returns
    -------
    True if obj is a Quantity with a scalar magnitude; False otherwise
    """
    return is_quantity(obj) and not iterable(obj._magnitude)


def is_quantity_with_sequence_magnitude(obj):
    """Test for Quantity with sequence magnitude.

    Parameters
    ----------
    obj : object


    Returns
    -------
    True if obj is a Quantity with a sequence magnitude; False otherwise

    Examples
    --------

    >>> is_quantity_with_sequence_magnitude([1, 2, 3])
    False

    >>> is_quantity_with_sequence_magnitude([1, Q_(2, 'm'), 3])
    False

    >>> is_quantity_with_sequence_magnitude(Q_([1, 2, 3], 'm'))
    True
    """
    return is_quantity(obj) and iterable(obj._magnitude)


def is_sequence_with_quantity_elements(obj):
    """Test for sequences of quantities.

    Parameters
    ----------
    obj : object


    Returns
    -------
    True if obj is a sequence and at least one element is a Quantity; False otherwise

    Examples
    --------

    >>> is_sequence_with_quantity_elements([1, 2, 3])
    False

    >>> is_sequence_with_quantity_elements([1, Q_(2, 'm'), 3])
    True

    >>> is_sequence_with_quantity_elements(Q_([1, 2, 3], 'm'))
    True
    """
    if np is not None and isinstance(obj, np.ndarray) and not obj.dtype.hasobject:
        # If obj is a numpy array, avoid looping on all elements
        # if dtype does not have objects
        return False
    return (
        iterable(obj)
        and sized(obj)
        and not isinstance(obj, str)
        and any(is_quantity(item) for item in obj)
    )


def is_quantity(obj):
    """Test for _units and _magnitude attrs.

    This is done in place of isinstance(Quantity, arg), which would cause a circular import.

    Parameters
    ----------
    obj : Object


    Returns
    -------
    bool
    """
    return hasattr(obj, "_units") and hasattr(obj, "_magnitude")
