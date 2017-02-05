"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import *
from cvxpy.error import DCPError
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
        new_obj, constraints = obj.canonical_form
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
        new_obj, constraints = obj.canonical_form
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

    def test_add_problems(self):
        """Test adding objectives.
        """
        expr1 = self.x**2
        expr2 = (self.x)**(-1)
        alpha = 2

        # Addition.

        assert (Minimize(expr1) + Minimize(expr2)).is_dcp()

        assert (Maximize(-expr1) + Maximize(-expr2)).is_dcp()

        # Test Minimize + Maximize
        with self.assertRaises(DCPError) as cm:
            Minimize(expr1) + Maximize(-expr2)
        self.assertEqual(str(cm.exception), "Problem does not follow DCP rules.")

        assert (Minimize(expr1) - Maximize(-expr2)).is_dcp()

        # Multiplication (alpha is a positive scalar).

        assert (alpha*Minimize(expr1)).is_dcp()

        assert (alpha*Maximize(-expr1)).is_dcp()

        assert (-alpha*Maximize(-expr1)).is_dcp()

        assert (-alpha*Maximize(-expr1)).is_dcp()
