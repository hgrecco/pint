"""
    pint.facets.numpy
    ~~~~~~~~~~~~~~~~~

    Adds pint the capability to interoperate with Dask

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

import functools

from ...compat import compute, dask_array, persist, visualize
from ..plain import PlainRegistry


def check_dask_array(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if isinstance(self._magnitude, dask_array.Array):
            return f(self, *args, **kwargs)
        else:
            msg = "Method {} only implemented for objects of {}, not {}".format(
                f.__name__, dask_array.Array, self._magnitude.__class__
            )
            raise AttributeError(msg)

    return wrapper


class DaskQuantity:

    # Dask.array.Array ducking
    def __dask_graph__(self):
        if isinstance(self._magnitude, dask_array.Array):
            return self._magnitude.__dask_graph__()
        else:
            return None

    def __dask_keys__(self):
        return self._magnitude.__dask_keys__()

    def __dask_tokenize__(self):
        from dask.base import tokenize

        from pint import UnitRegistry

        # TODO: Check if this is the right class as first argument
        return (UnitRegistry.Quantity, tokenize(self._magnitude), self.units)

    @property
    def __dask_optimize__(self):
        return dask_array.Array.__dask_optimize__

    @property
    def __dask_scheduler__(self):
        return dask_array.Array.__dask_scheduler__

    def __dask_postcompute__(self):
        func, args = self._magnitude.__dask_postcompute__()
        return self._dask_finalize, (func, args, self.units)

    def __dask_postpersist__(self):
        func, args = self._magnitude.__dask_postpersist__()
        return self._dask_finalize, (func, args, self.units)

    @staticmethod
    def _dask_finalize(results, func, args, units):
        values = func(results, *args)

        from pint import Quantity

        # TODO: Check if this is the right class as first argument
        return Quantity(values, units)

    @check_dask_array
    def compute(self, **kwargs):
        """Compute the Dask array wrapped by pint.PlainQuantity.

        Parameters
        ----------
        **kwargs : dict
            Any keyword arguments to pass to ``dask.compute``.

        Returns
        -------
        pint.PlainQuantity
            A pint.PlainQuantity wrapped numpy array.
        """
        (result,) = compute(self, **kwargs)
        return result

    @check_dask_array
    def persist(self, **kwargs):
        """Persist the Dask Array wrapped by pint.PlainQuantity.

        Parameters
        ----------
        **kwargs : dict
            Any keyword arguments to pass to ``dask.persist``.

        Returns
        -------
        pint.PlainQuantity
            A pint.PlainQuantity wrapped Dask array.
        """
        (result,) = persist(self, **kwargs)
        return result

    @check_dask_array
    def visualize(self, **kwargs):
        """Produce a visual representation of the Dask graph.

        The graphviz library is required.

        Parameters
        ----------
        **kwargs : dict
            Any keyword arguments to pass to ``dask.visualize``.

        Returns
        -------

        """
        visualize(self, **kwargs)


class DaskRegistry(PlainRegistry):

    _quantity_class = DaskQuantity
