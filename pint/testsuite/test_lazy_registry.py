"""Readiness and retry behavior of lazy registries."""

from __future__ import annotations

import copy
import pickle
import threading

import pytest

import pint
from pint.facets.plain.registry import GenericPlainRegistry
from pint.registry import ApplicationRegistry, LazyRegistry


@pytest.mark.parametrize("second_operation", ["unpickle", "quantity", "call"])
def test_concurrent_readers_wait_for_default_definitions(
    monkeypatch, func_registry, second_operation
):
    payload = pickle.dumps(func_registry.Quantity(7, "second"))
    registry = LazyRegistry()
    monkeypatch.setattr(pint, "application_registry", ApplicationRegistry(registry))
    loading = threading.Event()
    release = threading.Event()
    attempted = threading.Event()
    finished = threading.Event()
    results = {}
    errors = {}
    load_definitions = GenericPlainRegistry.load_definitions
    load_count = 0

    def pause_definitions(self, *args, **kwargs):
        nonlocal load_count
        if self is registry:
            load_count += 1
            loading.set()
            if not release.wait(5):
                raise TimeoutError("Initialization was not released")
        return load_definitions(self, *args, **kwargs)

    def unpickle(name):
        try:
            if name == "first" or second_operation == "unpickle":
                results[name] = pickle.loads(payload)
            elif second_operation == "quantity":
                results[name] = registry.Quantity(7, "second")
            else:
                results[name] = registry("7 second")
        except BaseException as error:
            errors[name] = error
        finally:
            if name == "second":
                finished.set()

    monkeypatch.setattr(GenericPlainRegistry, "load_definitions", pause_definitions)
    first = threading.Thread(target=unpickle, args=("first",), daemon=True)
    second = threading.Thread(target=unpickle, args=("second",), daemon=True)
    try:
        first.start()
        assert loading.wait(5)
        initializing_type = type(registry)
        lookup = initializing_type.__getattribute__

        def observe_lookup(self, name):
            if self is registry and threading.current_thread() is second:
                attempted.set()
            return lookup(self, name)

        monkeypatch.setattr(initializing_type, "__getattribute__", observe_lookup)
        second.start()
        assert attempted.wait(5)
        # Give the second reader a bounded opportunity to encounter the paused
        # initialization; a synchronized reader remains blocked until release.
        finished.wait(0.5)
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == {}
    assert set(results) == {"first", "second"}
    for result in results.values():
        assert result.magnitude == 7
        assert str(result.units) == "second"
        assert result._REGISTRY is registry
    assert load_count == 1


@pytest.mark.parametrize(
    "entry", ["__getattr__", "__call__", "__getitem__", "__setattr__"]
)
def test_saved_lazy_entry_uses_completed_registry(entry):
    registry = LazyRegistry()
    saved_entry = object.__getattribute__(registry, entry)
    assert registry.Quantity(7, "second").magnitude == 7

    if entry == "__getattr__":
        assert saved_entry("Quantity")(7, "second")._REGISTRY is registry
    elif entry == "__setattr__":
        saved_entry("case_sensitive", False)
        assert registry.case_sensitive is False
    else:
        if entry == "__getitem__":
            with pytest.warns(DeprecationWarning):
                result = saved_entry("7 second")
        else:
            result = saved_entry("7 second")
        assert result.magnitude == 7
        assert str(result.units) == "second"
        assert result._REGISTRY is registry


@pytest.mark.parametrize("subclass", [False, True])
def test_failed_initialization_retries_without_mutating_arguments(
    monkeypatch, subclass
):
    class CustomLazyRegistry(LazyRegistry):
        pass

    registry_type = CustomLazyRegistry if subclass else LazyRegistry
    preprocessors = [str.strip]
    kwargs = {"preprocessors": preprocessors}
    registry = registry_type(kwargs=kwargs)
    load_definitions = GenericPlainRegistry.load_definitions
    attempts = 0

    def fail_once(self, *args, **kwargs):
        nonlocal attempts
        if self is registry:
            attempts += 1
            if attempts == 1:
                raise ValueError("definition loading failed")
        return load_definitions(self, *args, **kwargs)

    monkeypatch.setattr(GenericPlainRegistry, "load_definitions", fail_once)
    with pytest.raises(ValueError, match="definition loading failed"):
        registry.Quantity(7, "second")
    assert type(registry) is registry_type
    result = registry.Quantity(7, "second")
    assert result.magnitude == 7
    assert str(result.units) == "second"
    assert result._REGISTRY is registry
    assert attempts == 2
    assert preprocessors == [str.strip]
    assert kwargs == {"preprocessors": [str.strip]}


def test_completed_lazy_registry_can_be_deepcopied():
    registry = LazyRegistry()
    registry.Quantity(7, "second")
    cloned = copy.deepcopy(registry)
    result = cloned.Quantity(7, "second")
    assert type(registry) is pint.UnitRegistry
    assert type(cloned) is pint.UnitRegistry
    assert result.magnitude == 7
    assert str(result.units) == "second"
    assert result._REGISTRY is cloned
    assert registry.Quantity(7, "second")._REGISTRY is registry


def _run_during_initialization(first_action, second_action, loading, release):
    """Capture thread exceptions and always release/join bounded test workers."""
    attempted = threading.Event()
    finished = threading.Event()
    results = {}
    errors = {}

    def run(name, action):
        try:
            if name == "second":
                attempted.set()
            results[name] = action()
        except BaseException as error:
            errors[name] = error
        finally:
            if name == "second":
                finished.set()

    first = threading.Thread(target=run, args=("first", first_action), daemon=True)
    second = threading.Thread(target=run, args=("second", second_action), daemon=True)
    try:
        first.start()
        assert loading.wait(5)
        second.start()
        assert attempted.wait(5)
        finished_before_release = finished.wait(0.5)
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive()
    assert not second.is_alive()
    return results, errors, finished_before_release


def test_waiting_reader_retries_after_initialization_failure(monkeypatch):
    registry = LazyRegistry()
    loading = threading.Event()
    release = threading.Event()
    load_definitions = GenericPlainRegistry.load_definitions
    attempts = 0

    def fail_once(self, *args, **kwargs):
        nonlocal attempts
        if self is registry:
            attempts += 1
            if attempts == 1:
                loading.set()
                if not release.wait(5):
                    raise TimeoutError("Initialization was not released")
                raise ValueError("definition loading failed")
        return load_definitions(self, *args, **kwargs)

    monkeypatch.setattr(GenericPlainRegistry, "load_definitions", fail_once)
    results, errors, _ = _run_during_initialization(
        lambda: registry.Quantity(7, "second"),
        lambda: registry.Quantity(7, "second"),
        loading,
        release,
    )
    assert set(errors) == {"first"}
    assert isinstance(errors["first"], ValueError)
    assert str(errors["first"]) == "definition loading failed"
    assert results["second"].magnitude == 7
    assert str(results["second"].units) == "second"
    assert results["second"]._REGISTRY is registry
    assert attempts == 2


def test_concurrent_write_is_not_overwritten_by_initialization(monkeypatch):
    registry = LazyRegistry()
    loading = threading.Event()
    release = threading.Event()
    initialize = pint.UnitRegistry.__init__

    def pause_constructor(self, *args, **kwargs):
        if self is registry:
            loading.set()
            if not release.wait(5):
                raise TimeoutError("Initialization was not released")
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(pint.UnitRegistry, "__init__", pause_constructor)
    results, errors, _ = _run_during_initialization(
        lambda: registry.Quantity(7, "second"),
        lambda: setattr(registry, "case_sensitive", False),
        loading,
        release,
    )
    assert errors == {}
    assert registry.case_sensitive is False
    assert results["first"]._REGISTRY is registry


def test_independent_lazy_registry_can_initialize_while_another_waits(monkeypatch):
    registry = LazyRegistry()
    independent = LazyRegistry()
    loading = threading.Event()
    release = threading.Event()
    load_definitions = GenericPlainRegistry.load_definitions

    def pause_definitions(self, *args, **kwargs):
        if self is registry:
            loading.set()
            if not release.wait(5):
                raise TimeoutError("Initialization was not released")
        return load_definitions(self, *args, **kwargs)

    monkeypatch.setattr(GenericPlainRegistry, "load_definitions", pause_definitions)
    results, errors, finished_before_release = _run_during_initialization(
        lambda: registry.Quantity(7, "second"),
        lambda: independent.Quantity(7, "second"),
        loading,
        release,
    )
    assert errors == {}
    assert finished_before_release
    assert results["first"]._REGISTRY is registry
    assert results["second"]._REGISTRY is independent


def test_initializing_thread_can_reenter_its_registry(monkeypatch):
    registry = LazyRegistry()
    after_init = GenericPlainRegistry._after_init
    callback_results = []

    def use_initialized_definitions(self):
        after_init(self)
        if self is registry:
            self.case_sensitive = False
            callback_results.append(self.Quantity(7, "second"))

    monkeypatch.setattr(
        GenericPlainRegistry, "_after_init", use_initialized_definitions
    )
    results = {}
    errors = []

    def initialize():
        try:
            results["quantity"] = registry.Quantity(7, "second")
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=initialize, daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert errors == []
    assert results["quantity"]._REGISTRY is registry
    assert len(callback_results) == 1
    assert callback_results[0].magnitude == 7
    assert str(callback_results[0].units) == "second"
    assert callback_results[0]._REGISTRY is registry
    assert registry.case_sensitive is False


@pytest.mark.parametrize(
    "retry_error",
    [AttributeError("retry load failed"), pint.UndefinedUnitError("missing")],
)
def test_waiting_reader_preserves_initialization_attribute_error(
    monkeypatch, retry_error
):
    registry = LazyRegistry()
    loading = threading.Event()
    release = threading.Event()
    entered = threading.Event()
    load_definitions = GenericPlainRegistry.load_definitions
    attempts = 0
    errors = {}

    def fail_loading(self, *args, **kwargs):
        nonlocal attempts
        if self is registry:
            attempts += 1
            if attempts == 1:
                loading.set()
                if not release.wait(5):
                    raise TimeoutError("Initialization was not released")
                raise ValueError("first load failed")
            if attempts == 2:
                raise retry_error
        return load_definitions(self, *args, **kwargs)

    def read(name):
        try:
            registry.Quantity(7, "second")
        except BaseException as error:
            errors[name] = error

    monkeypatch.setattr(GenericPlainRegistry, "load_definitions", fail_loading)
    first = threading.Thread(target=read, args=("first",), daemon=True)
    waiter = threading.Thread(target=read, args=("waiter",), daemon=True)
    try:
        first.start()
        assert loading.wait(5)
        initializing_type = type(registry)
        lookup = initializing_type.__getattribute__

        def observe(self, name):
            if (
                self is registry
                and threading.current_thread() is waiter
                and name == "Quantity"
            ):
                entered.set()
            return lookup(self, name)

        monkeypatch.setattr(initializing_type, "__getattribute__", observe)
        waiter.start()
        assert entered.wait(5)
    finally:
        release.set()
        first.join(5)
        if waiter.ident is not None:
            waiter.join(5)
    assert not first.is_alive()
    assert not waiter.is_alive()
    assert set(errors) == {"first", "waiter"}
    assert isinstance(errors["first"], ValueError)
    assert errors["waiter"] is retry_error
    assert attempts == 2


def test_deepcopy_during_final_publication_has_regular_registry_type():
    registry = LazyRegistry()
    ready = threading.Event()
    release = threading.Event()
    errors = []

    class PausePublication(dict):
        def __delitem__(self, key):
            super().__delitem__(key)
            if key == "_initialization_lock":
                ready.set()
                if not release.wait(5):
                    raise TimeoutError("Final publication was not released")

    object.__setattr__(
        registry,
        "__dict__",
        PausePublication(object.__getattribute__(registry, "__dict__")),
    )

    def initialize():
        try:
            registry.Quantity(7, "second")
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=initialize, daemon=True)
    try:
        worker.start()
        assert ready.wait(5)
        cloned = copy.deepcopy(registry)
        assert type(cloned) is pint.UnitRegistry
        result = cloned.Quantity(7, "second")
        assert result.magnitude == 7
        assert str(result.units) == "second"
        assert result._REGISTRY is cloned
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert errors == []
