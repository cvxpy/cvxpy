from cvxpy.expressions.variable import Variable
import unittest

class TestConstraints(unittest.TestCase):
    """ Unit tests for the expression/expression module. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

    # Test the EqConstraint class.
    def test_eq_constraint(self):
        constr = self.x == self.z
        self.assertEqual(constr.name(), "x == z")
        self.assertEqual(constr.size(), (2,1))
        self.assertEqual(constr.variables().keys(), ['x','z'])
        
        with self.assertRaises(Exception) as cm:
            (self.x == self.y).size()
        self.assertEqual(str(cm.exception), "'x == y' has incompatible dimensions.")

    # Test the LeqConstraint class.
    def test_leq_constraint(self):
        constr = self.x <= self.z
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.size(), (2,1))
        self.assertEqual(constr.variables().keys(), ['x','z'])
        
        with self.assertRaises(Exception) as cm:
            (self.x <= self.y).size()
        self.assertEqual(str(cm.exception), "'x <= y' has incompatible dimensions.")