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
