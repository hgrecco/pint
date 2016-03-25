# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function, absolute_import

from pint.compat import Loc
from pint.testsuite import helpers, BaseTestCase
from pint import UnitRegistry
import os

class TestBabel(BaseTestCase):

    @helpers.requires_proper_babel()
    def test_babel(self):
        ureg = UnitRegistry()
        dirname = os.path.dirname(__file__)
        ureg.load_definitions(os.path.join(dirname, '../xtranslated.txt'))
        locale = Loc('fr', 'FR')

        distance = 24.0 * ureg.meter
        self.assertEqual(
            distance.format_babel(locale=locale, form='long'),
            "24.0 mètres"
        )
        time = 8.0 * ureg.second
        self.assertEqual(
            time.format_babel(locale=locale, form='long'),
            "8.0 secondes"
        )
        velocity = distance / time ** 2
        self.assertEqual(
            velocity.format_babel(locale=locale, form='long'),
            "0.375 mètre par seconde²"
        )
        mks = ureg.get_system('mks')
        self.assertEqual(
            mks.format_babel(locale=locale),
            "métrique"
        )
