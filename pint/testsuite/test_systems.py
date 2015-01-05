# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import


from pint.systems import Group

from pint.testsuite import QuantityTestCase


class TestGroup(QuantityTestCase):

    def _build_default(self):
        d = Group('default', parent=None)
        return d

    def test_units_programatically(self):
        g1 = self._build_default()
        self.assertEqual(g1._child_groups, set())
        self.assertEqual(g1._parent_groups, set())
        g1.add_units('meter', 'second', 'meter')
        self.assertEqual(g1._unit_names, set(['meter', 'second']))
        self.assertEqual(g1.members, set(['meter', 'second']))

    def test_groups_programatically(self):
        g1 = self._build_default()
        g2 = Group('g2', parent=g1)

        self.assertEqual(g1._child_groups, set(['g2']))
        self.assertEqual(g1._parent_groups, set())

        self.assertEqual(g2._child_groups, set())
        self.assertEqual(g2._parent_groups, set(['default']))


    def test_simple(self):
        lines = ['@group mygroup',
                 'meter',
                 'second',
                 ]

        d = self._build_default()
        grp = Group.from_lines(lines, lambda x: None, d)

        self.assertEqual(grp.name, 'mygroup')
        self.assertEqual(grp._unit_names, set(['meter', 'second']))
        self.assertEqual(grp._child_groups, set())
        self.assertEqual(grp._parent_groups, set([d.name]))
        self.assertEqual(grp.members, frozenset(['meter', 'second']))

    def test_using1(self):
        lines = ['@group mygroup using group1',
                 'meter',
                 'second',
                 ]
        d = self._build_default()
        Group('group1', d)
        grp = Group.from_lines(lines, lambda x: None, d)
        self.assertEqual(grp.name, 'mygroup')
        self.assertEqual(grp._unit_names, set(['meter', 'second']))
        self.assertEqual(grp._child_groups, set(['group1']))
        self.assertEqual(grp.members, frozenset(['meter', 'second']))

    def test_using2(self):
        lines = ['@group mygroup using group1,group2',
                 'meter',
                 'second',
                 ]
        d = self._build_default()
        Group('group1', d)
        Group('group2', d)
        grp = Group.from_lines(lines, lambda x: None, d)
        self.assertEqual(grp.name, 'mygroup')
        self.assertEqual(grp._unit_names, set(['meter', 'second']))
        self.assertEqual(grp._child_groups, set(['group1', 'group2']))
        self.assertEqual(grp.members, frozenset(['meter', 'second']))

    def test_spaces(self):
        lines = ['@group   mygroup   using   group1 , group2',
                 ' meter ',
                 ' second  ',
                 ]
        d = self._build_default()
        Group('group1', d)
        Group('group2', d)
        grp = Group.from_lines(lines, lambda x: None, d)
        self.assertEqual(grp.name, 'mygroup')
        self.assertEqual(grp._unit_names, set(['meter', 'second']))
        self.assertEqual(grp._child_groups, set(['group1', 'group2']))
        self.assertEqual(grp.members, frozenset(['meter', 'second']))

    def test_invalidate_members(self):
        lines = ['@group mygroup using group1',
                 'meter',
                 'second',
                 ]
        d = self._build_default()
        g1 = Group('group1', d)
        grp = Group.from_lines(lines, lambda x: None, d)
        self.assertIs(d._computed_members, None)
        self.assertIs(grp._computed_members, None)
        self.assertEqual(grp.members, frozenset(['meter', 'second']))
        self.assertIs(d._computed_members, None)
        self.assertIsNot(grp._computed_members, None)
        self.assertEqual(d.members, frozenset(['meter', 'second']))
        self.assertIsNot(d._computed_members, None)
        self.assertIsNot(grp._computed_members, None)
        grp.invalidate_members()
        self.assertIs(d._computed_members, None)
        self.assertIs(grp._computed_members, None)

    def test_members_including(self):
        d = self._build_default()
        g1 = Group('group1', d)
        g1.add_units('second', 'inch')
        g2 = Group('group2', d)
        g2.add_units('second', 'newton')
        g3 = Group('group3', d)
        g3.add_units('meter', 'second')
        g3.add_groups('group1', 'group2')
        self.assertEqual(d.members, frozenset(['meter', 'second', 'newton', 'inch']))
        self.assertEqual(g1.members, frozenset(['second', 'inch']))
        self.assertEqual(g2.members, frozenset(['second', 'newton']))
        self.assertEqual(g3.members, frozenset(['meter', 'second', 'newton', 'inch']))

