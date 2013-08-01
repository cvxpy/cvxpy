from cvxpy.expressions.variable import Variable
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
        
        with self.assertRaises(Exception) as cm:
            (self.x == self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

    # Test the LeqConstraint class.
    def test_leq_constraint(self):
        constr = self.x <= self.z
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.size, (2,1))
        # self.assertItemsEqual(constr.variables().keys(), [self.x.id, self.z.id])
        
        with self.assertRaises(Exception) as cm:
            (self.x <= self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

    # Test the SOC class.
    def test_soc_constraint(self):
        exp = self.x + self.z
        scalar_exp = self.a + self.b
        constr = SOC(exp, scalar_exp)
        self.assertEqual(constr.size, 3)
        self.assertEqual(len(constr.format()), 2)

    # Test the SDC class.
    def test_sdc_constraint(self):
        exp = self.x + self.z
        scalar_exp = self.a + self.b
        constr = SOC(exp, scalar_exp)
        self.assertEqual(constr.size, 3)
        self.assertEqual(len(constr.format()), 2)
