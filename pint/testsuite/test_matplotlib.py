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


def test_set_ticks_before_axis_units_are_known(local_registry):
    # matplotlib passes a None unit while the axis has no unit yet (#1927)
    fig, ax = plt.subplots()

    ax.set_yticks(local_registry.Quantity([1.0, 2.0], "pascal"))
    ax.set_xticks([1.0 * local_registry.meter, 2.0 * local_registry.meter])

    assert ax.get_yticks() == pytest.approx([1.0, 2.0])
    assert ax.get_xticks() == pytest.approx([1.0, 2.0])


def test_axis_units_can_be_reset_to_none(local_registry):
    # plotting attaches the converter and sets the axis units (#1927)
    fig, ax = plt.subplots()
    ax.plot(
        local_registry.Quantity([1.0, 2.0], "second"),
        local_registry.Quantity([3.0, 4.0], "pascal"),
    )

    ax.yaxis.set_units(None)

    assert ax.yaxis.get_units() is None


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
