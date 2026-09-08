.. _numpy-implementation:

Adding support for numpy functions
==================================

The numpy support in Pint lives in ``pint/facets/numpy/numpy_func.py``.

Numpy function implementations
------------------------------

The basic way to implement a numpy function is through the ``@implements``
decorator. Its syntax is::

    @implements(numpy_func_string, func_type)
    def _some_func(arg1, arg2, ...):
        ...

where ``numpy_func_string`` is the name of the numpy function to support and
``func_type`` is either ``"function"`` (a numpy function) or ``"ufuncs"`` (a
numpy array method). The function replaces the original numpy function. Inside
it, you should do the following:

- Check that the input arguments meet the requirements, and convert the units
  of same-kind physical quantities to the unit of the first argument that has
  units. (For example, before computing ``1 m + 2 cm``, it converts first to
  ``1 m + 0.02 m``.)
- Use the original numpy function to compute on the *magnitude* of the input
  arguments.
- Create an output quantity object whose magnitude is the numpy output value
  and whose units are derived from the input units, then return it.

#### Individual implementations

For complex functions, a separate implementation must be written for each
function. For example::

    @implements("geomspace", "function")
    def _geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
        if all(not _is_quantity(arg) for arg in (start, stop)):
            return np.geomspace(start, stop, num, endpoint, dtype, axis)
        first_input_units = _get_first_input_units((start, stop))
        if not _is_quantity(start):
            start = start * first_input_units._REGISTRY.parse_units("dimensionless")
        if not _is_quantity(stop):
            stop = stop * first_input_units._REGISTRY.parse_units("dimensionless")

        start = _base_unit_if_needed(start)
        stop = _base_unit_if_needed(stop)
        (start, stop), output_wrap = unwrap_and_wrap_consistent_units(start, stop)
        return output_wrap(np.geomspace(start, stop, num, endpoint, dtype, axis))

### Shared implementations

If several functions have the same argument format and unit conversion
relationship, they can share an implementation. For example::

    def implement_mul_func(func):
        # If NumPy is not available, do not attempt implement that which does not exist
        if np is None:
            return
        if "." not in func_str:
            func = getattr(np, func_str, None)
        else:
            parts = func_str.split(".")
            module = np
            for part in parts[:-1]:
                module = getattr(module, part, None)
            func = getattr(module, parts[-1], None)

        # if NumPy does not implement it, do not implement it either
        if func is None:
            return

        @implements(func_str, "function")
        def implementation(a, b, **kwargs):
            a, b = _dimensionless_if_needed(a, b)
            a = _base_unit_if_needed(a)
            b = _base_unit_if_needed(b)
            units = a.units * b.units
            mag = func(a._magnitude, b._magnitude, **kwargs)
            return mag * units

The following are the commonly used shared implementations:

#### implement_consistent_units_by_argument

The output uses the unit of one of the input arguments, or requires multiple
input arguments to be same-kind physical quantities — the output then uses that
unit, or is unitless.

Its definition is::

    implement_consistent_units_by_argument(func_str, unit_arguments, wrap_output=True)

where:

- ``func_str``: the numpy function name to support (such as ``"mean"``,
  ``"clip"``).
- ``unit_arguments``: identifies which arguments of the function are
  unit-carrying physical quantities. It can be a single argument name (a
  string) or a list of argument names (``["start", "stop"]``).
- ``wrap_output``: whether to re-wrap the output with the input units. When
  ``False``, the output is unitless (such as ``searchsorted``).

#### strip_unit_input_output_ufuncs

Ignore the units of the inputs; the output is a unitless bare value. The
results of these ufuncs are independent of units, such as testing whether the
input is zero, finite, its sign bit, and so on.

#### matching_input_bare_output_ufuncs

Require all inputs to be same-kind physical quantities (converted to the unit
of the first input), but the output is a unitless bare value. These ufuncs are
usually comparisons, returning booleans that carry no physical meaning.

#### set_units_ufuncs

Specify the input and output units with a dict, ``(in_unit, out_unit)``: the
input is first converted to ``in_unit``, then its magnitude is handed to the
numpy ufunc for computation, and the output is wrapped with ``out_unit``.
Trigonometric functions (input in radians), inverse trigonometric functions
(output in radians), and exponential/logarithmic functions (input and output
dimensionless) belong to this category. ::

    set_units_ufuncs = {
        "cumprod": ("", ""),
        "arccos": ("", "radian"),   # inverse trig: unitless in, radians out
        "exp": ("", ""),            # exp/log: unitless in and out
        "sin": ("radian", ""),      # trig: radians in, unitless out
        "radians": ("degree", "radian"),
        "degrees": ("radian", "degree"),
        "deg2rad": ("degree", "radian"),
        "rad2deg": ("radian", "degree"),
        ...
    }

Implementation::

    for ufunc_str, (in_unit, out_unit) in set_units_ufuncs.items():
        implement_func("ufunc", ufunc_str, input_units=in_unit, output_unit=out_unit)

Note: an ``in_unit`` of ``""`` (the empty string) means dimensionless; an
``out_unit`` of ``""`` means the output is dimensionless.

#### matching_input_copy_units_output_ufuncs

Require all inputs to be same-kind physical quantities, and the output uses the
unit of the first input. These ufuncs do not change the physical meaning of the
input quantities, such as ``max``, ``mean``, ``min``, ``round``, ``hypot``, and
so on.

#### copy_units_output_ufuncs

Ignore the units of all inputs except the first, and the output uses the unit
of the first input. Used for ``ldexp``, ``fmod``, ``mod``, ``remainder``, and so
on. For these ufuncs, the second argument carries no physical unit of its own
(such as the modulus of a modulo operation), or its result is independent of
the second argument's units.

#### op_units_output_ufuncs

The output unit is determined from the input units by some operation
``unit_op``; see ``get_op_output_unit`` for the specific operations. For
example, ``multiply``'s output unit is the product of the input units
(``"mul"``), and ``sqrt``'s output unit is the square root of the input unit
(``"sqrt"``). ::

    op_units_output_ufuncs = {
        "var": "variance",     # variance: unit squared
        "multiply": "mul",     # multiply: units multiply
        "true_divide": "div",  # divide: units divide
        "divide": "div",
        "floor_divide": "div",
        "sqrt": "sqrt",        # square root
        "cbrt": "cbrt",        # cube root
        "square": "square",    # square
        "reciprocal": "reciprocal",  # reciprocal
        "std": "delta",        # standard deviation: uses the delta unit
        "sum": "sum",
        "cumsum": "sum",
        "cumulative_sum": "sum",
        "matmul": "mul",       # matrix multiplication: units multiply
    }

Implementation::

    for ufunc_str, unit_op in op_units_output_ufuncs.items():
        implement_func("ufunc", ufunc_str, input_units=None, output_unit=unit_op)

#### implement_func

This is the foundation on which all the above behaviors are built. Its
definition is::

    implement_func(func_type, func_str, input_units=None, output_unit=None)

- ``func_type``: ``"function"`` (a numpy function) or ``"ufunc"`` (a numpy
  ufunc).
- ``func_str``: the numpy function name to support (supports submodules, such
  as ``"linalg.norm"``).
- ``input_units``: controls how input arguments are converted to magnitudes
  before being handed to numpy.

  - ``pint.Unit``: first converts all arguments and keyword arguments to this
    unit, then takes their magnitudes.
  - ``"all_consistent"``: converts all arguments and keyword arguments to the
    unit of the first unit-carrying Quantity.
  - any other string: parsed as a unit, and all arguments are converted to it.
  - ``None``: no unit conversion; only strips the units (takes magnitudes).

- ``output_unit``: controls the unit of the output.

  - ``pint.Unit``: wraps the output with the specified unit.
  - ``"match_input"``: wraps the output with the unit of the first
    unit-carrying Quantity.
  - a ``unit_op`` string: derives the output unit from the input units via
    ``get_op_output_unit``.
  - any other string: parsed as a unit, used as the output unit.
  - ``None``: returns a bare magnitude, without units.

#### implement_mul_func

Provides a shared implementation for the several numpy functions whose "units
multiply", such as ``cross``, ``dot``, ``inner``, ``outer``, ``tensordot``,
``convolve``, ``matvec``, and so on. The output unit of these functions is the
product of the input argument units.

#### implement_solve_func

Similar to ``implement_mul_func``, but the output unit is the quotient of the
input units (``b.units / a.units``), used for solving linear systems, such as
``linalg.solve``, ``linalg.tensorsolve``, ``linalg.lstsq``. For the equation
``A·x = b``, the solution ``x`` has the units of ``b`` divided by the units of
``A``.

#### implement_consistent_units_by_argument

Performs unit consistency conversion on the specified arguments, and the output
uses the input units. Its definition is::

    implement_consistent_units_by_argument(func_str, unit_arguments, wrap_output=True)

- ``func_str``: the numpy function name to support (such as ``"mean"``,
  ``"clip"``, ``"linspace"``).
- ``unit_arguments``: identifies which arguments of the function are
  unit-carrying physical quantities. It can be a single argument name (a
  string) or a list of argument names (``["start", "stop"]``).
- ``wrap_output``: whether to re-wrap the output with the input units. When
  ``False``, the output is unitless (such as ``searchsorted``).

The implementation skips the specified arguments whose value is ``None`` (such
as ``clip``'s ``a_min``/``a_max`` when ``None``), and only performs consistency
conversion on the non-``None`` unit-carrying arguments.

#### Other shared implementations

- ``implement_close``: implements ``isclose`` and ``allclose``. If ``atol`` is a
  bare value, it is treated as a tolerance in the unit of the first input
  ``a``.
- ``implement_atleast_nd``: implements ``atleast_1d``, ``atleast_2d``,
  ``atleast_3d``, each output using the unit of its respective original input.
- ``implement_single_dimensionless_argument_func``: implements ``cumprod`` and
  ``nancumprod``. A cumulative product is only unit-consistent for dimensionless
  quantities, so the input is first converted to dimensionless.
- ``implement_prod_func``: implements ``prod`` and ``nanprod``. The output unit
  is determined by the number of elements (dimensions) participating in the
  cumulative product, i.e. ``a.units ** dimension``.
- ``implement_eq_ne_ufunc``: implements ``equal`` and ``not_equal``. Unlike the
  remaining comparison ufuncs, when dimensions are incompatible it should not
  raise an error, but follow Python's ``==``/``!=`` semantics and return
  ``False``/``True``.
- Beyond the above sets, ``implement_mul_func``, ``implement_solve_func`` and
  other functions that use the ``@implements`` decorator also have individual
  implementations for special cases (such as ``_add``, ``_subtract``,
  ``_modf``, ``_frexp``, ``_power``, and so on).

#### get_op_output_unit

When the ``output_unit`` of ``op_units_output_ufuncs`` and ``implement_func`` is
a ``unit_op`` string, this function derives the output unit from the input
units. Its signature is::

    get_op_output_unit(unit_op, first_input_units, all_args=None, size=None)

The supported ``unit_op`` values and their corresponding unit operations:

- ``"sum"``: uses ``first_input_units``; raises ``OffsetUnitCalculusError`` if
  it is not a multiplicative unit.
- ``"mul"``: the product of all argument units (multiplication).
- ``"delta"``: the delta version (difference) of ``first_input_units``; uses the
  delta unit for non-multiplicative units.
- ``"delta,div"``: same as ``"delta"``, but divided by the units of the remaining
  arguments.
- ``"div"``: the first argument's unit divided by the remaining arguments'
  units.
- ``"variance"``: the square of ``first_input_units``; raises
  ``OffsetUnitCalculusError`` if it is not a multiplicative unit.
- ``"square"``: the square of ``first_input_units``.
- ``"sqrt"``: the square root of ``first_input_units``.
- ``"cbrt"``: the cube root of ``first_input_units``.
- ``"reciprocal"``: the reciprocal of ``first_input_units``.
- ``"size"``: ``first_input_units`` raised to the ``size`` power.
- ``"invdiv"``: the inverse of ``"div"``; the product of the remaining arguments'
  units divided by the first argument's unit.

Among these, the ``unit_op`` values that involve multiplication, division,
squaring, or roots (``"mul"``, ``"div"``, ``"square"``, ``"sqrt"``, ``"cbrt"``,
``"reciprocal"``, ``"variance"``, and so on) all call ``_validated_muldiv_unit``
to check for non-multiplicative units, because non-multiplicative units are not
allowed to participate in multiplication and division.

Notes
-----

### Functions with multiple arguments are not necessarily all unit-carrying physical quantities

When implementing support for a function with multiple arguments, consider the
case where some user-supplied arguments are unit-carrying physical quantities
and others are bare numpy arrays or bare Python values. Bare values should be
treated as dimensionless quantities in the computation. You can use
``_dimensionless_if_needed`` to assign the dimensionless unit of the unit system
to bare values.

### Consider whether the function mathematically allows non-multiplicative units

In Pint, temperature is represented by non-multiplicative units, such as degrees
Celsius ``degC`` and degrees Fahrenheit ``degF``. Non-multiplicative units have a
nonzero offset.

Mathematically speaking, whether a function can support non-multiplicative units
depends on whether it is an **offset-preserving** affine function.

Let the true physical quantity (base unit) of temperature be :math:`T`, the
value expressed in a non-multiplicative unit be :math:`t`, the offset be
:math:`c`, and the scaling factor be :math:`a`; then we have the affine
relationship:

.. math::

    T = a\, t + c

For example, for Celsius :math:`a = 1,\ c = 273.15`; for Fahrenheit
:math:`a = 5/9,\ c = 255.37\dots`.

If a function :math:`f` acts on the value :math:`t`, we want computing
:math:`f(t)` under a non-multiplicative unit to be equivalent to first
converting to the base unit, computing :math:`f(T)`, and converting back; that
is, we require :math:`f` to satisfy:

.. math::

    f\big(a\,t + c\big) = a\, f(t) + c

That is, :math:`f` must be a **linear (or affine) operation that preserves the
offset**, such as:

- Translation (adding a constant difference measured in the corresponding delta
  unit): :math:`t \mapsto t + \Delta`
- Averaging: :math:`\frac{1}{n}\sum t_i`, because
  :math:`\frac{1}{n}\sum (a t_i + c) = a\frac{1}{n}\sum t_i + c`
- Taking the max/min: :math:`\max_i(a t_i + c) = a\max_i t_i + c`
- Order-preserving statistics such as the median and quantiles

These functions all satisfy :math:`f(T) = a\,f(t) + c`, so their results can
still be correctly represented with the original non-multiplicative unit.

Conversely, functions that **do not** satisfy the above property cannot support
non-multiplicative units, including:

- ``sum``, ``cumsum``, ``cumulative_sum``: :math:`\sum (a t_i + c) = a\sum t_i + n
  c \ne a\sum t_i + c`; the offset is multiplied by the element count :math:`n`,
  so the original unit can no longer be used.
- Multiplication, division, powers, and inner products such as ``multiply``,
  ``divide``, ``sqrt``, ``square``, ``reciprocal``, ``power``, ``dot``, ``cross``,
  and so on: the offset of a non-multiplicative unit cannot participate in
  multiplication, so the non-multiplicative unit must first be converted to the
  base unit.

It can be summarized in one sentence: **only "offset-preserving affine
transformations" can use a non-multiplicative unit as input or output; any
function involving addition with repeated counting, or non-affine operations
such as multiplication, division, or powers, breaks offset preservation and
therefore does not support non-multiplicative units.**

In Pint's concrete implementation, this is reflected in:

- ``"sum"`` raises ``OffsetUnitCalculusError`` when it encounters a
  non-multiplicative unit.
- ``"delta"`` (and ``std``, which uses ``"delta"``), and ``cov`` use the delta
  unit (temperature difference) rather than temperature itself when they
  encounter a non-multiplicative unit, thereby expressing it correctly.
- All operations involving multiplication or division (``"mul"``, ``"div"``,
  ``"square"``, ``"sqrt"``, ``"cbrt"``, ``"reciprocal"``, and so on) are checked
  via ``_validated_muldiv_unit`` and raise ``OffsetUnitCalculusError`` when they
  encounter a non-multiplicative unit.
- Offset-preserving statistics such as ``mean``, ``max``, ``min``, ``median`` use
  the ``"match_input"`` output unit, directly reusing the input unit, and so
  support non-multiplicative units.
- Functions such as ``isnan``, ``isinf``, ``size`` operate independently of
  units, so they support non-multiplicative units.

If a user still needs to run a function that does not support non-multiplicative
units on Celsius temperature, they should take ``.magnitude`` themselves and
then call the function.
