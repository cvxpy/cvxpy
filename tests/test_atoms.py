from cvxpy.atoms.abs import abs
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
import unittest

class TestAtoms(unittest.TestCase):
    """ Unit tests for the atoms module. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

    # Test the abs class.
    def test_abs(self):
        exp = self.x+self.y
        atom = abs(exp)
        self.assertEquals(atom.name(), "abs(x + y)")
        self.assertEquals(atom.size(), (1,1))
        self.assertEquals(atom.curvature(), Curvature.CONVEX)

        obj,constraints = atom.canonicalize()
        names = [c.name() for c in constraints]
        expected = [(exp <= obj).name(), (-obj <= exp).name()]
        self.assertItemsEqual(names, expected)