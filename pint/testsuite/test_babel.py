# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function, absolute_import

from pint.compat import Loc
from pint.testsuite import helpers, BaseTestCase
from pint import UnitRegistry

class TestBabel(BaseTestCase):

    @helpers.requires_proper_babel()
    def test_babel(self):
        ureg = UnitRegistry()
        ureg.load_definitions('../xtranslated.txt')
        locale = Loc('fr', 'FR')

        distance = 24.0 * ureg.meter
        self.assertEqual(
            distance.format_babel(locale=locale, form='long'),
            u'24.0 mètres'
        )
        time = 8.0 * ureg.second
        self.assertEqual(
            time.format_babel(locale=locale, form='long'),
            u'8.0 secondes'
        )
        velocity = distance / time ** 2
        self.assertEqual(
            velocity.format_babel(locale=locale, form='long'),
            u'0.375 mètre par seconde²'
        )
