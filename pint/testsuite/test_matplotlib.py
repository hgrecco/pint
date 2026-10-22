from __future__ import annotations

import pytest

from pint import UnitRegistry

# Conditionally import matplotlib and NumPy
plt = pytest.importorskip("matplotlib.pyplot", reason="matplotlib is not available")
np = pytest.importorskip("numpy", reason="NumPy is not available")


@pytest.fixture(scope="module")
def local_registry():
    # Set up unit registry for matplotlib
    ureg = UnitRegistry()
    ureg.setup_matplotlib(True)
    return ureg


# Set up matplotlib
plt.switch_backend("agg")


def test_plot_empty_quantities(local_registry):
    fig, ax = plt.subplots()
    try:
        (line,) = ax.plot([] * local_registry.seconds, [] * local_registry.meters)
        assert ax.xaxis.units == local_registry.seconds
        assert ax.yaxis.units == local_registry.meters
        assert line.get_xdata(orig=False).size == 0
        assert line.get_ydata(orig=False).size == 0
    finally:
        plt.close(fig)


def test_empty_ticks_without_axis_units(local_registry):
    fig, ax = plt.subplots()
    try:
        ax.set_yticks([] * local_registry.pascal, minor=True)
        assert ax.yaxis.units is None
        assert ax.get_yticks(minor=True).size == 0
    finally:
        plt.close(fig)


def test_empty_quantity_conversion_respects_explicit_units(local_registry):
    from pint.errors import DimensionalityError
    from pint.matplotlib import PintConverter

    converter = PintConverter(local_registry)
    empty = [] * local_registry.meter
    assert converter.convert(empty, local_registry.centimeter, None).size == 0
    with pytest.raises(DimensionalityError):
        converter.convert(empty, local_registry.second, None)


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_basic_plot(local_registry):
    y = np.linspace(0, 30) * local_registry.miles
    x = np.linspace(0, 5) * local_registry.hours

    fig, ax = plt.subplots()
    ax.plot(x, y, "tab:blue")
    ax.axhline(26400 * local_registry.feet, color="tab:red")
    ax.axvline(120 * local_registry.minutes, color="tab:green")

    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_plot_with_set_units(local_registry):
    y = np.linspace(0, 30) * local_registry.miles
    x = np.linspace(0, 5) * local_registry.hours

    fig, ax = plt.subplots()
    ax.yaxis.set_units(local_registry.inches)
    ax.xaxis.set_units(local_registry.seconds)

    ax.plot(x, y, "tab:blue")
    ax.axhline(26400 * local_registry.feet, color="tab:red")
    ax.axvline(120 * local_registry.minutes, color="tab:green")

    return fig


@pytest.mark.mpl_image_compare(tolerance=0, remove_text=True)
def test_plot_with_non_default_format(local_registry):
    local_registry.mpl_formatter = "{:~P}"

    y = np.linspace(0, 30) * local_registry.miles
    x = np.linspace(0, 5) * local_registry.hours

    fig, ax = plt.subplots()
    ax.yaxis.set_units(local_registry.inches)
    ax.xaxis.set_units(local_registry.seconds)

    ax.plot(x, y, "tab:blue")
    ax.axhline(26400 * local_registry.feet, color="tab:red")
    ax.axvline(120 * local_registry.minutes, color="tab:green")

    return fig
