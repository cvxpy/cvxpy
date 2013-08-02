from cvxpy.atoms import *
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import *
import unittest

class TestObjectives(unittest.TestCase):
    """ Unit tests for the expression/expression module. """
    def setUp(self):
        self.x = Variable(name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(name='z')

    # Test the Minimize class.
    def test_minimize(self):
        exp = self.x + self.z
        obj = Minimize(exp)
        self.assertEqual(obj.name(), "minimize %s" % exp.name())
        new_obj,constraints = obj.canonicalize()
        #self.assertEqual(constraints[0].name(), (new_obj == exp).name())
        self.assertEqual(len(constraints), 1)

        with self.assertRaises(Exception) as cm:
            Minimize(self.y).canonicalize()
        self.assertEqual(str(cm.exception), 
            "The objective 'minimize y' must resolve to a scalar.")

    # Test the Maximize class.
    def test_maximize(self):
        exp = self.x + self.z
        obj = Maximize(exp)
        self.assertEqual(obj.name(), "maximize %s" % exp.name())
        new_obj,constraints = obj.canonicalize()
        #self.assertEqual(constraints[0].name(), (new_obj == exp).name())
        self.assertEqual(len(constraints), 1)

        with self.assertRaises(Exception) as cm:
            Maximize(self.y).canonicalize()
        self.assertEqual(str(cm.exception), 
            "The objective 'maximize y' must resolve to a scalar.")

    # Test is_dcp for Minimize and Maximize
    def test_is_dcp(self):
        self.assertEqual(Minimize(normInf(self.x)).is_dcp(), True)
        self.assertEqual(Minimize(-normInf(self.x)).is_dcp(), False)

        self.assertEqual(Maximize(normInf(self.x)).is_dcp(), False)
        self.assertEqual(Maximize(-normInf(self.x)).is_dcp(), True)