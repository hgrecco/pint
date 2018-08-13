import pandas as pd


def delegated_method(method, index, name, *args, **kwargs):
    return pd.Series(method(*args, **kwargs), index, name=name)


class Delegated:
    # Descriptor for delegating attribute access to from
    # a Series to an underlying array

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, '_index')
        name = object.__getattribute__(obj, '_name')
        result = self._get_result(obj)
        return pd.Series(result, index, name=name)


class DelegatedProperty(Delegated):
    def _get_result(self, obj, type=None):
        return getattr(object.__getattribute__(obj, '_data'), self.name)


class DelegatedMethod(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, '_index')
        name = object.__getattribute__(obj, '_name')
        method = getattr(object.__getattribute__(obj, '_data'), self.name)
        return delegated_method(method, index, name)
