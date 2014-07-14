"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.expressions.variables import Variable
from cvxpy.constraints.second_order import SOC
import unittest

class TestConstraints(unittest.TestCase):
    """ Unit tests for the expression/expression module. """
    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    # Test the EqConstraint class.
    def test_eq_constraint(self):
        constr = self.x == self.z
        self.assertEqual(constr.name(), "x == z")
        self.assertEqual(constr.size, (2,1))
        # self.assertItemsEqual(constr.variables().keys(), [self.x.id, self.z.id])
        # Test value and dual_value.
        assert constr.dual_value is None
        assert constr.value is None
        self.x.save_value(2)
        self.z.save_value(2)
        assert constr.value
        self.x.save_value(3)
        assert not constr.value

        with self.assertRaises(Exception) as cm:
            (self.x == self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 1) (3, 1)")

    # Test the LeqConstraint class.
    def test_leq_constraint(self):
        constr = self.x <= self.z
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.size, (2,1))
        # Test value and dual_value.
        assert constr.dual_value is None
        assert constr.value is None
        self.x.save_value(1)
        self.z.save_value(2)
        assert constr.value
        self.x.save_value(3)
        assert not constr.value
        # self.assertItemsEqual(constr.variables().keys(), [self.x.id, self.z.id])

        with self.assertRaises(Exception) as cm:
            (self.x <= self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 1) (3, 1)")

    # Test the SOC class.
    def test_soc_constraint(self):
        exp = self.x + self.z
        scalar_exp = self.a + self.b
        constr = SOC(scalar_exp, [exp])
        self.assertEqual(constr.size, (3,1))

    def test_chained_constraints(self):
        """Tests that chaining constraints raises an error.
        """
        with self.assertRaises(Exception) as cm:
            (self.z <= self.x <= 1)
        self.assertEqual(str(cm.exception), "Cannot evaluate the truth value of a constraint.")

        with self.assertRaises(Exception) as cm:
            (self.x == self.z == 1)
        self.assertEqual(str(cm.exception), "Cannot evaluate the truth value of a constraint.")
