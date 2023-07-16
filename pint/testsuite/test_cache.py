# This is a weird test module as it is currently testing python's cache
# Its purpose is to summarize the requirements for any replacements
# and test for undocumented features.

from pint.cache import cache


class Demo:
    def __init__(self, value) -> None:
        self.value = value

    @cache
    def calculated_value(self, value):
        return self.value * value


class DerivedDemo(Demo):
    @cache
    def calculated_value(self, value):
        if value is None:
            return super().calculated_value(3)
        return self.value * value + 0.5


def test_cache_clear():
    demo = Demo(2)

    assert demo.calculated_value(3) == 6
    assert demo.calculated_value(3) == 6
    demo.value = 3
    assert demo.calculated_value(3) == 6
    demo.calculated_value.cache_clear(demo)
    assert demo.calculated_value(3) == 9
    assert demo.calculated_value(3) == 9


def test_per_instance_cache():
    demo2 = Demo(2)
    demo3 = Demo(3)

    assert demo2.calculated_value(3) == 6
    assert demo2.calculated_value(3) == 6
    assert demo3.calculated_value(3) == 9
    assert demo3.calculated_value(3) == 9


def test_per_instance_cache_clear():
    demo2 = Demo(2)
    demo3 = Demo(3)

    demo2.calculated_value(3)
    demo3.calculated_value(3)

    demo2.value = 4
    demo3.value = 5
    assert demo2.calculated_value(3) == 6
    assert demo3.calculated_value(3) == 9
    demo2.calculated_value.cache_clear(demo2)
    assert demo2.calculated_value(3) == 12
    assert demo3.calculated_value(3) == 9
    demo3.calculated_value.cache_clear(demo3)
    assert demo3.calculated_value(5) == 15


def test_inheritance():
    demo = DerivedDemo(2)

    assert demo.calculated_value(3) == 6.5
    assert demo.calculated_value(3) == 6.5
    assert demo.calculated_value(None) == 6
    assert demo.calculated_value(None) == 6
    assert demo.calculated_value(1)
