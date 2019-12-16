import itertools

from pint.compat import np
from pint.converters import Converter, OffsetConverter, ScaleConverter
from pint.testsuite import BaseTestCase, helpers


class TestConverter(BaseTestCase):
    def test_converter(self):
        c = Converter()
        self.assertTrue(c.is_multiplicative)
        self.assertTrue(c.to_reference(8))
        self.assertTrue(c.from_reference(8))

    def test_multiplicative_converter(self):
        c = ScaleConverter(20.0)
        self.assertEqual(c.from_reference(c.to_reference(100)), 100)
        self.assertEqual(c.to_reference(c.from_reference(100)), 100)

    def test_offset_converter(self):
        c = OffsetConverter(20.0, 2)
        self.assertEqual(c.from_reference(c.to_reference(100)), 100)
        self.assertEqual(c.to_reference(c.from_reference(100)), 100)

    @helpers.requires_numpy()
    def test_converter_inplace(self):
        for c in (ScaleConverter(20.0), OffsetConverter(20.0, 2)):
            fun1 = lambda x, y: c.from_reference(c.to_reference(x, y), y)
            fun2 = lambda x, y: c.to_reference(c.from_reference(x, y), y)
            for fun, (inplace, comp) in itertools.product(
                (fun1, fun2), ((True, self.assertIs), (False, self.assertIsNot))
            ):
                a = np.ones((1, 10))
                ac = np.ones((1, 10))
                r = fun(a, inplace)
                np.testing.assert_allclose(r, ac)
                comp(a, r)
