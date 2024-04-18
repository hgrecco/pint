from __future__ import annotations

import pytest

from pint import UnitRegistry


class TestKind:
    def test_torque_energy(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        Kind = ureg.Kind
        moment_arm = Q_(1, "m").to_kind("[moment_arm]")
        force = Q_(1, "lbf").to_kind("[force]")
        # to_kind converts to the preferred_unit of the kind
        assert force.units == ureg.N

        # both force and moment_arm have kind defined.
        # Torque is defined in default_en:
        # [torque] = [force] * [moment_arm]
        # the result is a quantity with kind [torque]
        torque = force * moment_arm
        assert torque.units == ureg.N * ureg.m
        assert torque.kinds == Kind("[torque]")

        # Energy is defined in default_en:
        # [energy] = [force] * [length] = J
        distance = Q_(1, "m").to_kind("[length]")
        energy = force * distance
        assert energy.kinds == Kind("[energy]")
        assert energy.units == ureg.J

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

    @pytest.mark.xfail
    def test_three_parameters(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        energy = Q_(1, "J").to_kind("[energy]")
        mass = Q_(1, "kg").to_kind("[mass]")
        temperature = Q_(1, "K").to_kind("[temperature]")

        # [specific_heat_capacity] = [energy] / [temperature] / [mass]
        specific_heat_capacity = energy / temperature / mass
        assert specific_heat_capacity.kind == "[specific_heat_capacity]"
