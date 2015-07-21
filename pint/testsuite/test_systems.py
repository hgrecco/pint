# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import


from pint import UnitRegistry
from pint.systems import Group, System

from pint.testsuite import QuantityTestCase


class TestGroup(QuantityTestCase):

    def _build_root(self, d):
        root = Group('root', d)
        return root

    def test_units_programatically(self):
        d = {}
        root = self._build_root(d)
        self.assertEqual(root._used_groups, set())
        self.assertEqual(root._used_by, set())
        root.add_units('meter', 'second', 'meter')
        self.assertEqual(root._unit_names, set(['meter', 'second']))
        self.assertEqual(root.members, set(['meter', 'second']))

        self.assertEqual(set(d.keys()), set(['root']))

    def test_cyclic(self):
        d = {}
        root = self._build_root(d)
        g2 = Group('g2', d)
        g3 = Group('g3', d)
        g2.add_groups('g3')

        self.assertRaises(ValueError, g2.add_groups, 'root')
        self.assertRaises(ValueError, g3.add_groups, 'g2')
        self.assertRaises(ValueError, g3.add_groups, 'root')

    def test_groups_programatically(self):
        d = {}
        root = self._build_root(d)
        g2 = Group('g2', d)

        self.assertEqual(set(d.keys()), set(['root', 'g2']))

        self.assertEqual(root._used_groups, set(['g2']))
        self.assertEqual(root._used_by, set())

        self.assertEqual(g2._used_groups, set())
        self.assertEqual(g2._used_by, set(['root']))


    def test_simple(self):
        lines = ['@group mygroup',
                 'meter',
                 'second',
                 ]

        d = {}
        root = self._build_root(d)

        grp = Group.from_lines(lines, lambda x: None, d)

        self.assertEqual(set(d.keys()), set(['root', 'mygroup']))

        self.assertEqual(grp.name, 'mygroup')
        self.assertEqual(grp._unit_names, set(['meter', 'second']))
        self.assertEqual(grp._used_groups, set())
        self.assertEqual(grp._used_by, set([root.name]))
        self.assertEqual(grp.members, frozenset(['meter', 'second']))

    def test_using1(self):
        lines = ['@group mygroup using group1',
                 'meter',
                 'second',
                 ]

        d = {}
        root = self._build_root(d)

        g = Group('group1', d)
        grp = Group.from_lines(lines, lambda x: None, d)
        self.assertEqual(grp.name, 'mygroup')
        self.assertEqual(grp._unit_names, set(['meter', 'second']))
        self.assertEqual(grp._used_groups, set(['group1']))
        self.assertEqual(grp.members, frozenset(['meter', 'second']))

    def test_using2(self):
        lines = ['@group mygroup using group1,group2',
                 'meter',
                 'second',
                 ]


        d = {}
        root = self._build_root(d)

        Group('group1', d)
        Group('group2', d)
        grp = Group.from_lines(lines, lambda x: None, d)
        self.assertEqual(grp.name, 'mygroup')
        self.assertEqual(grp._unit_names, set(['meter', 'second']))
        self.assertEqual(grp._used_groups, set(['group1', 'group2']))
        self.assertEqual(grp.members, frozenset(['meter', 'second']))

    def test_spaces(self):
        lines = ['@group   mygroup   using   group1 , group2',
                 ' meter ',
                 ' second  ',
                 ]


        d = {}
        root = self._build_root(d)

        Group('group1', d)
        Group('group2', d)
        grp = Group.from_lines(lines, lambda x: None, d)
        self.assertEqual(grp.name, 'mygroup')
        self.assertEqual(grp._unit_names, set(['meter', 'second']))
        self.assertEqual(grp._used_groups, set(['group1', 'group2']))
        self.assertEqual(grp.members, frozenset(['meter', 'second']))

    def test_invalidate_members(self):
        lines = ['@group mygroup using group1',
                 'meter',
                 'second',
                 ]

        d = {}
        root = self._build_root(d)

        g1 = Group('group1', d)
        grp = Group.from_lines(lines, lambda x: None, d)
        self.assertIs(root._computed_members, None)
        self.assertIs(grp._computed_members, None)
        self.assertEqual(grp.members, frozenset(['meter', 'second']))
        self.assertIs(root._computed_members, None)
        self.assertIsNot(grp._computed_members, None)
        self.assertEqual(root.members, frozenset(['meter', 'second']))
        self.assertIsNot(root._computed_members, None)
        self.assertIsNot(grp._computed_members, None)
        grp.invalidate_members()
        self.assertIs(root._computed_members, None)
        self.assertIs(grp._computed_members, None)

    def test_with_defintions(self):
        lines = ['@group imperial',
                 'inch',
                 'yard',
                 'kings_leg = 2 * meter',
                 'kings_head = 52 * inch'
                 'pint'
                 ]
        defs = []
        def define(ud):
            defs.append(ud.name)

        d = {}
        root = self._build_root(d)

        grp = Group.from_lines(lines, define, d)

        self.assertEqual(['kings_leg', 'kings_head'], defs)

    def test_members_including(self):

        d = {}
        root = self._build_root(d)

        g1 = Group('group1', d)

        g1.add_units('second', 'inch')
        g2 = Group('group2', d)
        g2.add_units('second', 'newton')

        g3 = Group('group3', d)
        g3.add_units('meter', 'second')
        g3.add_groups('group1', 'group2')

        self.assertEqual(root.members, frozenset(['meter', 'second', 'newton', 'inch']))
        self.assertEqual(g1.members, frozenset(['second', 'inch']))
        self.assertEqual(g2.members, frozenset(['second', 'newton']))
        self.assertEqual(g3.members, frozenset(['meter', 'second', 'newton', 'inch']))

    def test_get_compatible_units(self):
        ureg = UnitRegistry()

        g = ureg.get_group('imperial')
        g.add_units('inch', 'yard', 'pint')
        c = ureg.get_compatible_units('meter', 'imperial')
        self.assertEqual(c, frozenset([ureg.inch, ureg.yard]))



class TestSystem(QuantityTestCase):

    def _build_root(self, d):
        root = Group('root', d)
        return root

    def test_implicit_root(self):
        lines = ['@system mks',
                 'meter',
                 'kilogram',
                 'second',
                 ]
        d = {}
        root = self._build_root(d)
        s = System.from_lines(lines, lambda x: x, d)
        s._used_groups = set(['root'])

    def test_simple_using(self):
        lines = ['@system mks using g1',
                 'meter',
                 'kilogram',
                 'second',
                 ]
        d = {}
        root = self._build_root(d)
        s = System.from_lines(lines, lambda x: x, d)
        s._used_groups = set(['root', 'g1'])


    def test_members_group(self):
        lines = ['@system mk',
                 'meter',
                 'kilogram',
                 ]
        d = {}
        root = self._build_root(d)
        root.add_units('second')
        s = System.from_lines(lines, lambda x: x, d)
        self.assertEqual(s.members, frozenset(['second']))

    def test_get_compatible_units(self):
        sysname = 'mysys1'
        ureg = UnitRegistry()

        g = ureg.get_group('imperial')

        g.add_units('inch', 'yard', 'pint')
        c = ureg.get_compatible_units('meter', 'imperial')
        self.assertEqual(c, frozenset([ureg.inch, ureg.yard]))

        lines = ['@system %s using imperial' % sysname,
                 'inch',
                 ]

        s = System.from_lines(lines, lambda x: x, g._groups_systems)
        c = ureg.get_compatible_units('meter', sysname)
        self.assertEqual(c, frozenset([ureg.inch, ureg.yard]))

    def test_get_base_units(self):
        sysname = 'mysys2'

        ureg = UnitRegistry()

        g = ureg.get_group('imperial')
        g.add_units('inch', 'yard', 'pint')

        lines = ['@system %s using imperial' % sysname,
                 'inch',
                 ]

        s = System.from_lines(lines, ureg.get_base_units, g._groups_systems)

        # base_factor, destination_units
        c = ureg.get_base_units('inch', system=sysname)
        self.assertAlmostEqual(c[0], 1)
        self.assertEqual(c[1], {'inch': 1})

        c = ureg.get_base_units('cm', system=sysname)
        self.assertAlmostEqual(c[0], 1./2.54)
        self.assertEqual(c[1], {'inch': 1})

    def test_get_base_units_different_exponent(self):
        sysname = 'mysys3'

        ureg = UnitRegistry()

        g = ureg.get_group('imperial')
        g.add_units('inch', 'yard', 'pint')
        c = ureg.get_compatible_units('meter', 'imperial')

        lines = ['@system %s using imperial' % sysname,
                 'pint:meter',
                 ]

        s = System.from_lines(lines, ureg.get_base_units, g._groups_systems)

        # base_factor, destination_units
        c = ureg.get_base_units('inch', system=sysname)
        self.assertAlmostEqual(c[0], 0.326, places=3)
        self.assertEqual(c[1], {'pint': 1./3})

        c = ureg.get_base_units('cm', system=sysname)
        self.assertAlmostEqual(c[0], 0.1283, places=3)
        self.assertEqual(c[1], {'pint': 1./3})

        c = ureg.get_base_units('inch**2', system=sysname)
        self.assertAlmostEqual(c[0], 0.326**2, places=3)
        self.assertEqual(c[1], {'pint': 2./3})

        c = ureg.get_base_units('cm**2', system=sysname)
        self.assertAlmostEqual(c[0], 0.1283**2, places=3)
        self.assertEqual(c[1], {'pint': 2./3})

    def test_get_base_units_relation(self):
        sysname = 'mysys4'

        ureg = UnitRegistry()

        g = ureg.get_group('imperial')
        g.add_units('inch', 'yard', 'pint')

        lines = ['@system %s using imperial' % sysname,
                 'mph:meter',
                 ]

        s = System.from_lines(lines, ureg.get_base_units, g._groups_systems)

        # base_factor, destination_units
        c = ureg.get_base_units('inch', system=sysname)
        self.assertAlmostEqual(c[0], 0.0568, places=3)
        self.assertEqual(c[1], {'mph': 1, 'second': 1})

        c = ureg.get_base_units('kph', system=sysname)
        self.assertAlmostEqual(c[0], .6214, places=4)
        self.assertEqual(c[1], {'mph': 1})
