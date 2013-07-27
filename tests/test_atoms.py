from cvxpy.atoms import *
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
import cvxpy.interface.matrix_utilities as intf
import unittest

class TestAtoms(unittest.TestCase):
    """ Unit tests for the atoms module. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

    # Test the normInf class.
    def test_normInf(self):
        exp = self.x+self.y
        atom = normInf(exp)
        self.assertEquals(atom.name(), "normInf(x + y)")
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, Curvature.CONVEX)
        self.assertEquals(normInf(atom).curvature, Curvature.UNKNOWN)
        self.assertEquals(normInf(-atom).curvature, Curvature.UNKNOWN)

        with self.assertRaises(Exception) as cm:
            normInf([[1,2],[3,4]]).canonicalize()
        self.assertEqual(str(cm.exception), 
            "The argument '[[1, 2], [3, 4]]' to normInf must resolve to a vector.")

    # Test the norm1 class.
    def test_norm1(self):
        exp = self.x+self.y
        atom = norm1(exp)
        self.assertEquals(atom.name(), "norm1(x + y)")
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, Curvature.CONVEX)
        self.assertEquals(norm1(atom).curvature, Curvature.UNKNOWN)
        self.assertEquals(norm1(-atom).curvature, Curvature.UNKNOWN)

        with self.assertRaises(Exception) as cm:
            norm1([[1,2],[3,4]]).canonicalize()
        self.assertEqual(str(cm.exception), 
            "The argument '[[1, 2], [3, 4]]' to norm1 must resolve to a vector.")

    # Test the norm2 class.
    def test_norm2(self):
        exp = self.x+self.y
        atom = norm2(exp)
        self.assertEquals(atom.name(), "norm2(x + y)")
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, Curvature.CONVEX)
        self.assertEquals(norm2(atom).curvature, Curvature.UNKNOWN)
        self.assertEquals(norm2(-atom).curvature, Curvature.UNKNOWN)

        with self.assertRaises(Exception) as cm:
            norm2([[1,2],[3,4]]).canonicalize()
        self.assertEqual(str(cm.exception), 
            "The argument '[[1, 2], [3, 4]]' to norm2 must resolve to a vector.")