from __future__ import annotations

import pytest

from pint import UnitRegistry
from pint.util import UnitsContainer


class TestKind:
    def test_torque_energy(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        Kind = ureg.Kind
        QK_ = ureg.QuantityKind
        moment_arm = Q_(1, "m").to_kind("[moment_arm]")
        force = Q_(1, "lbf").to_kind("[force]")
        # to_kind converts to the preferred_unit of the kind
        assert force.units == ureg.N
        assert force == QK_(1, "[force]", "lbf")

        # both force and moment_arm have kind defined.
        # Torque is defined in default_en:
        # [torque] = [force] * [moment_arm]
        # the result is a quantity with kind [torque]
        torque = force * moment_arm
        assert torque.units == ureg.N * ureg.m
        assert torque.kinds == Kind("[torque]")

        assert force == torque / moment_arm
        assert moment_arm == torque / force

        # Energy is defined in default_en:
        # [energy] = [force] * [length] = J
        distance = Q_(1, "m").to_kind("[length]")
        energy = force * distance
        assert energy.kinds == Kind("[energy]")
        assert energy.units == ureg.J

        assert force == energy / distance
        assert distance == energy / force

        # Torque is not energy so cannot be added
        with pytest.raises(ValueError):
            energy + torque

    def test_acceleration(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        Kind = ureg.Kind
        velocity = Q_(1, "m/s").to_kind("[velocity]")
        time = Q_(1, "s").to_kind("[time]")
        acceleration = velocity / time
        assert acceleration.kinds == Kind("[acceleration]")
        # no preferred unit defined for acceleration, so uses base units
        assert acceleration.units == ureg.m / ureg.s**2

    def test_momentum(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        Kind = ureg.Kind
        mass = Q_(1, "kg").to_kind("[mass]")
        velocity = Q_(1, "m/s").to_kind("[velocity]")
        momentum = mass * velocity
        assert momentum.kinds == Kind("[momentum]")
        # no preferred unit defined for momentum, so uses base units
        # ensure gram is not used as base unit
        assert momentum.units == ureg.kg * ureg.m / ureg.s

    def test_compatible_kinds(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        q = Q_(1, "N m")
        assert "[torque]" in q.compatible_kinds()
        assert "[energy]" in q.compatible_kinds()

    @pytest.mark.xfail(reason="Not sure how to deal with order of operations")
    def test_three_parameters(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        energy = Q_(1, "J").to_kind("[energy]")
        mass = Q_(1, "kg").to_kind("[mass]")
        temperature = Q_(1, "K").to_kind("[temperature]")

        ureg.define("[specific_heat_capacity] = [energy] / [temperature] / [mass]")
        specific_heat_capacity = energy / mass / temperature
        assert specific_heat_capacity.kinds == "[specific_heat_capacity]"

        # this fails, giving specific_heat_capacity.kinds == [entropy] / [mass]
        specific_heat_capacity = energy / temperature / mass
        assert specific_heat_capacity.kinds == "[specific_heat_capacity]"

    def test_electrical_power(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        Kind = ureg.Kind

        ureg.define("[real_power] = [power] = W")
        ureg.define("[apparent_power] = [electric_potential] * [current] = VA")
        ureg.define(
            "[power_factor] = [real_power] / [apparent_power]"
        )  # don't have a way to set unit as dimensionless

        real_power = Q_(1, "W").to_kind("[real_power]")
        apparent_power = Q_(1, "VA").to_kind("[apparent_power]")
        power_factor = real_power / apparent_power
        assert power_factor.kinds == Kind("[power_factor]")
        assert power_factor.units == ureg.Unit("W / VA")

    def test_kindkind(self):
        ureg = UnitRegistry()
        Kind = ureg.Kind

        torque = Kind("[torque]")
        force = Kind("[force]")
        moment_arm = Kind("[moment_arm]")

        assert torque == force * moment_arm
        assert force == torque / moment_arm

        assert ureg.Unit("N") in force.compatible_units()

    kind_relations = [
        ("[length]", UnitsContainer({"[wavenumber]": -1.0})),
        ("[length]", UnitsContainer({"[area]": 0.5})),
        (
            "[length]",
            UnitsContainer({"[current]": 1.0, "[magnetic_field_strength]": -1.0}),
        ),
        ("[length]", UnitsContainer({"[charge]": -1.0, "[electric_dipole]": 1.0})),
        (
            "[length]",
            UnitsContainer({"[electric_field]": -1.0, "[electric_potential]": 1.0}),
        ),
        ("[length]", UnitsContainer({"[energy]": 1.0, "[force]": -1.0})),
        (
            "[length]",
            UnitsContainer({"[esu_charge]": 1.0, "[esu_electric_potential]": -1.0}),
        ),
        (
            "[length]",
            UnitsContainer(
                {"[esu_current]": 1.0, "[esu_magnetic_field_strength]": -1.0}
            ),
        ),
        (
            "[length]",
            UnitsContainer(
                {"[gaussian_charge]": 1.0, "[gaussian_electric_potential]": -1.0}
            ),
        ),
        (
            "[length]",
            UnitsContainer(
                {"[gaussian_charge]": -1.0, "[gaussian_electric_dipole]": 1.0}
            ),
        ),
        (
            "[length]",
            UnitsContainer(
                {
                    "[gaussian_electric_field]": -1.0,
                    "[gaussian_electric_potential]": 1.0,
                }
            ),
        ),
        (
            "[length]",
            UnitsContainer(
                {"[gaussian_resistance]": -1.0, "[gaussian_resistivity]": 1.0}
            ),
        ),
        ("[length]", UnitsContainer({"[moment_arm]": 1.0})),
        ("[length]", UnitsContainer({"[resistance]": -1.0, "[resistivity]": 1.0})),
        ("[length]", UnitsContainer({"[time]": 1.0, "[velocity]": 1.0})),
        (
            "[length]",
            UnitsContainer(
                {
                    "[esu_charge]": 0.6666666666666666,
                    "[mass]": -0.3333333333333333,
                    "[time]": 0.6666666666666666,
                }
            ),
        ),
        (
            "[length]",
            UnitsContainer(
                {
                    "[gaussian_charge]": 0.6666666666666666,
                    "[mass]": -0.3333333333333333,
                    "[time]": 0.6666666666666666,
                }
            ),
        ),
        ("[length]", UnitsContainer({"[volume]": 0.3333333333333333})),
    ]

    @pytest.mark.parametrize(("kind", "relation"), kind_relations)
    def test_kindkind_kind_relations(self, kind, relation):
        ureg = UnitRegistry()
        Kind = ureg.Kind
        kind = Kind(kind)
        assert relation in kind.kind_relations()

    def test_density(self):
        # https://github.com/hgrecco/pint/issues/676#issuecomment-689157693
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        Kind = ureg.Kind

        length = Q_(10, "m").to_kind("[length]")
        diameter = Q_(1, "mm").to_kind("[length]")
        rho = Q_(7200, "kg/m^3").to_kind("[density]")
        # mass calc from issue:
        # mass = rho * (ureg.pi / 4) * diameter**2 * length
        volume = (3.14159 / 4) * diameter**2 * length
        mass = rho * volume

        assert mass.kinds == Kind("[mass]")
        assert mass.units == ureg.kg

    def test_kindunit(self):
        # https://github.com/hgrecco/pint/issues/676
        ureg = UnitRegistry()
        # Kind = ureg.Kind
        newton = ureg.Unit("N")
        # assert Kind("[force]") in newton.compatible_kinds()
        assert "[force]" in newton.compatible_kinds()

    def test_waterpump(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        Kind = ureg.Kind
        shaft_speed = Q_(1200, "rpm").to_kind("[angular_frequency]")
        flow_rate = Q_(8.72, "m^3 / hr").to_kind("[volumetric_flow_rate]")
        pressure_delta = Q_(162, "kPa").to_kind("[pressure]")
        shaft_power = Q_(1.32, "kW").to_kind("[power]")
        # efficiency = Q_(30.6, "%").to_kind("[dimensionless]")

        shaft_torque = shaft_power / shaft_speed
        assert shaft_torque.kinds == Kind("[torque]")

        fluid_power = flow_rate * pressure_delta
        assert fluid_power.kinds == Kind("[power]")
