from __future__ import annotations

import pytest

from pint import DimensionalityError
from pint.testsuite import QuantityTestCase, helpers


# TODO: do not subclass from QuantityTestCase
@helpers.requires_not_uncertainties()
class TestNotMeasurement(QuantityTestCase):
    def test_instantiate(self):
        M_ = self.ureg.Measurement
        with pytest.raises(RuntimeError):
            M_(4.0, 0.1, "s")


# TODO: do not subclass from QuantityTestCase
@helpers.requires_uncertainties()
class TestMeasurement(QuantityTestCase):
    def test_simple(self):
        M_ = self.ureg.Measurement
        m = M_(4.0, 0.1, "s * s")
        assert repr(m) == "<Measurement(4.0, 0.1, second ** 2)>"

    def test_build(self):
        M_ = self.ureg.Measurement
        v, u = self.Q_(4.0, "s"), self.Q_(0.1, "s")
        M_(v.magnitude, u.magnitude, "s")
        ms = (
            M_(v.magnitude, u.magnitude, "s"),
            M_(v, u.magnitude),
            M_(v, u),
            v.plus_minus(0.1),
            v.plus_minus(0.025, True),
            v.plus_minus(u),
        )

        for m in ms:
            assert m.value == v
            assert m.error == u
            assert m.rel == m.error / abs(m.value)

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("", "(4.00 +/- 0.10) second ** 2"),
            ("P", "(4.00 ± 0.10) second²"),
            ("L", r"\left(4.00 \pm 0.10\right)\ \mathrm{second}^{2}"),
            ("H", "(4.00 &plusmn; 0.10) second<sup>2</sup>"),
            ("C", "(4.00+/-0.10) second**2"),
            ("Lx", r"\SI{4.00 +- 0.10}{\second\squared}"),
            (".1f", "(4.0 +/- 0.1) second ** 2"),
            (".1fP", "(4.0 ± 0.1) second²"),
            (".1fL", r"\left(4.0 \pm 0.1\right)\ \mathrm{second}^{2}"),
            (".1fH", "(4.0 &plusmn; 0.1) second<sup>2</sup>"),
            (".1fC", "(4.0+/-0.1) second**2"),
            (".1fLx", r"\SI{4.0 +- 0.1}{\second\squared}"),
        ],
    )
    def test_format(self, func_registry, spec, expected):
        Q_ = func_registry.Quantity
        v, u = Q_(4.0, "s ** 2"), Q_(0.1, "s ** 2")
        m = func_registry.Measurement(v, u)
        assert format(m, spec) == expected

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("uS", "0.200(10) second ** 2"),
            (".3uS", "0.2000(100) second ** 2"),
            (".3uSP", "0.2000(100) second²"),
            (".3uSL", r"0.2000\left(100\right)\ \mathrm{second}^{2}"),
            (".3uSH", "0.2000(100) second<sup>2</sup>"),
            (".3uSC", "0.2000(100) second**2"),
        ],
    )
    def test_format_paru(self, func_registry, spec, expected):
        Q_ = func_registry.Quantity
        v, u = Q_(0.20, "s ** 2"), Q_(0.01, "s ** 2")
        m = func_registry.Measurement(v, u)
        assert format(m, spec) == expected

    @pytest.mark.parametrize(
        "spec, expected",
        [
            (".3u", "(0.2000 +/- 0.0100) second ** 2"),
            (".3uP", "(0.2000 ± 0.0100) second²"),
            (".3uL", r"\left(0.2000 \pm 0.0100\right)\ \mathrm{second}^{2}"),
            (".3uH", "(0.2000 &plusmn; 0.0100) second<sup>2</sup>"),
            (".3uC", "(0.2000+/-0.0100) second**2"),
            (
                ".3uLx",
                r"\SI{0.2000 +- 0.0100}{\second\squared}",
            ),
            (".1uLx", r"\SI{0.20 +- 0.01}{\second\squared}"),
        ],
    )
    def test_format_u(self, func_registry, spec, expected):
        Q_ = func_registry.Quantity
        v, u = Q_(0.20, "s ** 2"), Q_(0.01, "s ** 2")
        m = func_registry.Measurement(v, u)
        assert format(m, spec) == expected

    @pytest.mark.parametrize(
        "spec, expected",
        [
            (".1u%", "(20 +/- 1)% second ** 2"),
            (".1u%P", "(20 ± 1)% second²"),
            (".1u%L", r"\left(20 \pm 1\right) \%\ \mathrm{second}^{2}"),
            (".1u%H", "(20 &plusmn; 1)% second<sup>2</sup>"),
            (".1u%C", "(20+/-1)% second**2"),
        ],
    )
    def test_format_percu(self, func_registry, spec, expected):
        Q_ = func_registry.Quantity
        v, u = Q_(0.20, "s ** 2"), Q_(0.01, "s ** 2")
        m = func_registry.Measurement(v, u)
        assert format(m, spec) == expected

    @pytest.mark.parametrize(
        "spec, expected",
        [
            (".1ue", "(2.0 +/- 0.1)e-01 second ** 2"),
            (".1ueP", "(2.0 ± 0.1)×10⁻¹ second²"),
            (
                ".1ueL",
                r"\left(2.0 \pm 0.1\right) \times 10^{-1}\ \mathrm{second}^{2}",
            ),
            (".1ueH", "(2.0 &plusmn; 0.1)×10<sup>-1</sup> second<sup>2</sup>"),
            (".1ueC", "(2.0+/-0.1)e-01 second**2"),
        ],
    )
    def test_format_perce(self, func_registry, spec, expected):
        Q_ = func_registry.Quantity
        v, u = Q_(0.20, "s ** 2"), Q_(0.01, "s ** 2")
        m = func_registry.Measurement(v, u)
        assert format(m, spec) == expected

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("", "(4.00 +/- 0.10)e+20 second ** 2"),
            # ("!r", "<Measurement(4e+20, 1e+19, second ** 2)>"),
            ("P", "(4.00 ± 0.10)×10²⁰ second²"),
            ("L", r"\left(4.00 \pm 0.10\right) \times 10^{20}\ \mathrm{second}^{2}"),
            ("H", "(4.00 &plusmn; 0.10)×10<sup>20</sup> second<sup>2</sup>"),
            ("C", "(4.00+/-0.10)e+20 second**2"),
            ("Lx", r"\SI{4.00 +- 0.10 e+20}{\second\squared}"),
        ],
    )
    def test_format_exponential_pos(self, func_registry, spec, expected):
        # Quantities in exponential format come with their own parenthesis, don't wrap
        # them twice
        m = func_registry.Quantity(4e20, "s^2").plus_minus(1e19)
        assert format(m, spec) == expected

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("", "(4.00 +/- 0.10)e-20 second ** 2"),
            # ("!r", "<Measurement(4e-20, 1e-21, second ** 2)>"),
            ("P", "(4.00 ± 0.10)×10⁻²⁰ second²"),
            (
                "L",
                r"\left(4.00 \pm 0.10\right) \times 10^{-20}\ \mathrm{second}^{2}",
            ),
            ("H", "(4.00 &plusmn; 0.10)×10<sup>-20</sup> second<sup>2</sup>"),
            ("C", "(4.00+/-0.10)e-20 second**2"),
            ("Lx", r"\SI{4.00 +- 0.10 e-20}{\second\squared}"),
        ],
    )
    def test_format_exponential_neg(self, func_registry, spec, expected):
        m = func_registry.Quantity(4e-20, "s^2").plus_minus(1e-21)
        assert format(m, spec) == expected

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("", "(4.00 +/- 0.10) second ** 2"),
            ("P", "(4.00 ± 0.10) second²"),
            ("L", r"\left(4.00 \pm 0.10\right)\ \mathrm{second}^{2}"),
            ("H", "(4.00 &plusmn; 0.10) second<sup>2</sup>"),
            ("C", "(4.00+/-0.10) second**2"),
            ("Lx", r"\SI{4.00 +- 0.10}{\second\squared}"),
            (".1f", "(4.0 +/- 0.1) second ** 2"),
            (".1fP", "(4.0 ± 0.1) second²"),
            (".1fL", r"\left(4.0 \pm 0.1\right)\ \mathrm{second}^{2}"),
            (".1fH", "(4.0 &plusmn; 0.1) second<sup>2</sup>"),
            (".1fC", "(4.0+/-0.1) second**2"),
            (".1fLx", r"\SI{4.0 +- 0.1}{\second\squared}"),
        ],
    )
    def test_format_default(self, func_registry, spec, expected):
        v, u = (
            func_registry.Quantity(4.0, "s ** 2"),
            func_registry.Quantity(0.1, "s ** 2"),
        )
        m = func_registry.Measurement(v, u)
        func_registry.default_format = spec
        assert f"{m}" == expected

    def test_raise_build(self):
        v, u = self.Q_(1.0, "s"), self.Q_(0.1, "s")
        o = self.Q_(0.1, "m")

        M_ = self.ureg.Measurement
        with pytest.raises(DimensionalityError):
            M_(v, o)
        with pytest.raises(DimensionalityError):
            v.plus_minus(o)
        with pytest.raises(ValueError):
            v.plus_minus(u, relative=True)

    def test_propagate_linear(self):
        v1, u1 = self.Q_(8.0, "s"), self.Q_(0.7, "s")
        v2, u2 = self.Q_(5.0, "s"), self.Q_(0.6, "s")
        v2, u3 = self.Q_(-5.0, "s"), self.Q_(0.6, "s")

        m1 = v1.plus_minus(u1)
        m2 = v2.plus_minus(u2)
        m3 = v2.plus_minus(u3)

        for factor, m in zip((3, -3, 3, -3), (m1, m3, m1, m3)):
            r = factor * m
            helpers.assert_quantity_almost_equal(
                r.value.magnitude, factor * m.value.magnitude
            )
            helpers.assert_quantity_almost_equal(
                r.error.magnitude, abs(factor * m.error.magnitude)
            )
            assert r.value.units == m.value.units

        for ml, mr in zip((m1, m1, m1, m3), (m1, m2, m3, m3)):
            r = ml + mr
            helpers.assert_quantity_almost_equal(
                r.value.magnitude, ml.value.magnitude + mr.value.magnitude
            )
            helpers.assert_quantity_almost_equal(
                r.error.magnitude,
                (
                    ml.error.magnitude + mr.error.magnitude
                    if ml is mr
                    else (ml.error.magnitude**2 + mr.error.magnitude**2) ** 0.5
                ),
            )
            assert r.value.units == ml.value.units

        for ml, mr in zip((m1, m1, m1, m3), (m1, m2, m3, m3)):
            r = ml - mr
            helpers.assert_quantity_almost_equal(
                r.value.magnitude, ml.value.magnitude - mr.value.magnitude
            )
            helpers.assert_quantity_almost_equal(
                r.error.magnitude,
                0
                if ml is mr
                else (ml.error.magnitude**2 + mr.error.magnitude**2) ** 0.5,
            )
            assert r.value.units == ml.value.units

    def test_propagate_product(self):
        v1, u1 = self.Q_(8.0, "s"), self.Q_(0.7, "s")
        v2, u2 = self.Q_(5.0, "s"), self.Q_(0.6, "s")
        v2, u3 = self.Q_(-5.0, "s"), self.Q_(0.6, "s")

        m1 = v1.plus_minus(u1)
        m2 = v2.plus_minus(u2)
        m3 = v2.plus_minus(u3)

        m4 = (2.3 * self.ureg.meter).plus_minus(0.1)
        m5 = (1.4 * self.ureg.meter).plus_minus(0.2)

        for ml, mr in zip((m1, m1, m1, m3, m4), (m1, m2, m3, m3, m5)):
            r = ml * mr
            helpers.assert_quantity_almost_equal(
                r.value.magnitude, ml.value.magnitude * mr.value.magnitude
            )
            assert r.value.units == ml.value.units * mr.value.units

        for ml, mr in zip((m1, m1, m1, m3, m4), (m1, m2, m3, m3, m5)):
            r = ml / mr
            helpers.assert_quantity_almost_equal(
                r.value.magnitude, ml.value.magnitude / mr.value.magnitude
            )
            assert r.value.units == ml.value.units / mr.value.units

    def test_measurement_comparison(self):
        x = self.Q_(4.2, "meter")
        y = self.Q_(5.0, "meter").plus_minus(0.1)
        assert x <= y
        assert not (x >= y)

    def test_tokenization(self):
        from pint import pint_eval

        pint_eval.tokenizer = pint_eval.uncertainty_tokenizer
        for p in pint_eval.tokenizer("8 + / - 4"):
            print(p)
        assert True
