"""functools.py - Tools for working with functions and callable objects
"""
# Python module wrapper for _functools C module
# to allow utilities written in Python to be added
# to the functools module.
# Written by Nick Coghlan <ncoghlan at gmail.com>,
# Raymond Hettinger <python at rcn.com>,
# and Łukasz Langa <lukasz at langa.pl>.
#   Copyright (C) 2006-2013 Python Software Foundation.
# See C source code for _functools credits/copyright

from __future__ import annotations

__all__ = [
    "cache",
    "lru_cache",
]

import warnings
from weakref import WeakKeyDictionary

from functools import update_wrapper

from typing import Any, Callable, Protocol, TYPE_CHECKING, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from . import UnitRegistry


################################################################################
### LRU Cache function decorator
################################################################################


class Hashable(Protocol):
    def __hash__(self) -> int:
        ...


class _HashedSeq(list[Any]):
    """This class guarantees that hash() will be called no more than once
    per element.  This is important because the lru_cache() will hash
    the key multiple times on a cache miss.

    """

    __slots__ = "hashvalue"

    def __init__(self, tup: tuple[Any, ...], hashfun: Callable[[Any], int] = hash):
        self[:] = tup
        self.hashvalue = hashfun(tup)

    def __hash__(self) -> int:
        return self.hashvalue


def _make_key(
    args: tuple[Any, ...],
    kwds: dict[str, Any],
    kwd_mark: tuple[Any, ...] = (object(),),
    fasttypes: set[type] = {int, str},
    tuple: type = tuple,
    type: type = type,
    len: Callable[[Any], int] = len,
) -> Hashable:
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    """
    # All of code below relies on kwds preserving the order input by the user.
    # Formerly, we sorted() the kwds before looping.  The new way is *much*
    # faster; however, it means that f(x=1, y=2) will now be treated as a
    # distinct call from f(y=2, x=1) which will be cached separately.
    key = args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    if len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


class lru_cache:
    """Least-recently-used cache decorator.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(decimal.Decimal("3.0")) and f(3.0) will be treated as
    distinct calls with distinct results. Some types such as str and int may
    be cached separately even when typed is false.

    Arguments to the cached function must be hashable.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    See:  https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)

    """

    # Users should only access the lru_cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the lru_cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    def __init__(self, user_function):
        wrapper = _lru_cache_wrapper(user_function)
        self.wrapped_fun = update_wrapper(wrapper, user_function)

    def __set_name__(self, owner, name):
        cache_methods = getattr(owner, "cache_methods", None)
        if cache_methods is None:
            owner._cache_methods = cache_methods = []

        cache_methods.append(self.wrapped_fun)

        setattr(owner, name, self.wrapped_fun)


def _lru_cache_wrapper(user_function: Callable[..., T]) -> Callable[..., T]:
    # Constants shared by all lru cache instances:

    sentinel = object()  # unique object used to signal cache misses
    make_key = _make_key  # build a key from the function arguments

    cache: WeakKeyDictionary[object, dict[Any, T]] = WeakKeyDictionary()

    stack: WeakKeyDictionary[object, list[dict[Any, T]]] = WeakKeyDictionary()

    def wrapper(self: UnitRegistry, *args: Any, **kwds: Any) -> T:
        # Simple caching without ordering or size limit

        key = make_key(args, kwds)

        subcache = cache.get(self, None)
        if subcache is None:
            cache[self] = subcache = {}

        result = subcache.get(key, sentinel)

        if result is not sentinel:
            return result

        subcache[key] = result = user_function(self, *args, **kwds)
        return result

    def cache_clear(self: UnitRegistry):
        """Clear the cache and cache statistics"""
        if self in cache:
            cache[self].clear()

    def cache_stack_push(self: UnitRegistry, obj: dict[Any, T] | None = None) -> None:
        substack = stack.get(self, None)
        if substack is None:
            stack[self] = substack = []

        subcache = cache.get(self, None)
        if subcache is None:
            cache[self] = subcache = {}

        substack.append(subcache)
        cache[self] = obj or {}

    def cache_stack_pop(self: UnitRegistry) -> dict[Any, T]:
        substack = stack.get(self, None)
        if substack is None:
            stack[self] = substack = []

        subcache = cache.get(self, None)
        if subcache is None:
            cache[self] = subcache = {}

        if substack:
            cache[self] = substack.pop()
        else:
            warnings.warn("Cannot pop cache from stack: stack is empty.")

        return subcache

    wrapper.cache_clear = cache_clear
    wrapper.cache_stack_push = cache_stack_push
    wrapper.cache_stack_pop = cache_stack_pop

    return wrapper


################################################################################
### cache -- simplified access to the infinity cache
################################################################################


def cache(user_function: Callable[..., Any], /):
    'Simple lightweight unbounded cache.  Sometimes called "memoize".'
    return lru_cache(user_function)
