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

from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import *
import unittest

class TestObjectives(unittest.TestCase):
    """ Unit tests for the expression/expression module. """
    def setUp(self):
        self.x = Variable(name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(name='z')

    def test_str(self):
        """Test string representations.
        """
        obj = Minimize(self.x)
        self.assertEqual(repr(obj), "Minimize(%s)" % repr(self.x))
        obj = Minimize(2*self.x)
        self.assertEqual(repr(obj), "Minimize(%s)" % repr(2*self.x))

        obj = Maximize(self.x)
        self.assertEqual(repr(obj), "Maximize(%s)" % repr(self.x))
        obj = Maximize(2*self.x)
        self.assertEqual(repr(obj), "Maximize(%s)" % repr(2*self.x))

    # Test the Minimize class.
    def test_minimize(self):
        exp = self.x + self.z
        obj = Minimize(exp)
        self.assertEqual(str(obj), "minimize %s" % exp.name())
        new_obj,constraints = obj.canonical_form
        #self.assertEqual(constraints[0].name(), (new_obj == exp).name())
        # for affine objectives, there should be no constraints
        self.assertEqual(len(constraints), 0)

        with self.assertRaises(Exception) as cm:
            Minimize(self.y).canonical_form
        self.assertEqual(str(cm.exception),
            "The 'minimize' objective must resolve to a scalar.")

        # Test copy with args=None
        copy = obj.copy()
        self.assertTrue(type(copy) is type(obj))
        # A new object is constructed, so copy.args == obj.args but copy.args
        # is not obj.args.
        self.assertEqual(copy.args, obj.args)
        self.assertFalse(copy.args is obj.args)
        # Test copy with new args
        copy = obj.copy(args=[square(self.z)])
        self.assertTrue(type(copy) is type(obj))
        self.assertTrue(copy.args[0].args[0] is self.z)

    # Test the Maximize class.
    def test_maximize(self):
        exp = self.x + self.z
        obj = Maximize(exp)
        self.assertEqual(str(obj), "maximize %s" % exp.name())
        new_obj,constraints = obj.canonical_form
        #self.assertEqual(constraints[0].name(), (new_obj == exp).name())
        # for affine objectives, there should be no constraints
        self.assertEqual(len(constraints), 0)

        with self.assertRaises(Exception) as cm:
            Maximize(self.y).canonical_form
        self.assertEqual(str(cm.exception),
            "The 'maximize' objective must resolve to a scalar.")

        # Test copy with args=None
        copy = obj.copy()
        self.assertTrue(type(copy) is type(obj))
        # A new object is constructed, so copy.args == obj.args but copy.args
        # is not obj.args.
        self.assertEqual(copy.args, obj.args)
        self.assertFalse(copy.args is obj.args)
        # Test copy with new args
        copy = obj.copy(args=[-square(self.x)])
        self.assertTrue(type(copy) is type(obj))
        self.assertTrue(copy.args[0].args[0].args[0] is self.x)

    # Test is_dcp for Minimize and Maximize
    def test_is_dcp(self):
        self.assertEqual(Minimize(normInf(self.x)).is_dcp(), True)
        self.assertEqual(Minimize(-normInf(self.x)).is_dcp(), False)

        self.assertEqual(Maximize(normInf(self.x)).is_dcp(), False)
        self.assertEqual(Maximize(-normInf(self.x)).is_dcp(), True)
