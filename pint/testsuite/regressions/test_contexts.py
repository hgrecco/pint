from pint import UnitRegistry


class TestRegressions:

    def test_lng_conversion(self):
        ureg = UnitRegistry()
        ureg.load_definitions('pint/testsuite/regressions/conv.txt')
        ureg.enable_contexts("lng")
        u  = ureg.Unit("watt")
        assert u.compatible_units()
