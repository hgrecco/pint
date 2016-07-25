from pint import UnitRegistry, set_application_registry
from pint.testsuite import QuantityTestCase

ureg = UnitRegistry()
set_application_registry(ureg)
Q = ureg.Quantity

class TestInferBaseUnit(QuantityTestCase):
    def test_infer_base_unit(self):
        from pint.util import infer_base_unit
        self.assertEqual(infer_base_unit(Q(1, 'millimeter * nanometer')), Q(1, 'meter**2').units)

    def test_to_compact(self):
        r = Q(1000000000, 'm') * Q(1, 'mm') / Q(1, 's') / Q(1, 'ms')
        compact_r = r.to_compact()
        expected = Q(1000., 'kilometer**2 / second**2')
        self.assertQuantityAlmostEqual(compact_r, expected)

