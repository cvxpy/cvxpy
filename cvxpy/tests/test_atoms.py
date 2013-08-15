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
from cvxpy.expressions.variable import Variable
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
import unittest

class TestAtoms(unittest.TestCase):
    """ Unit tests for the atoms module. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    # Test the normInf class.
    def test_normInf(self):
        exp = self.x+self.y
        atom = normInf(exp)
        self.assertEquals(atom.name(), "normInf(x + y)")
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX)
        self.assertEquals(normInf(atom).curvature, u.Curvature.UNKNOWN)
        self.assertEquals(normInf(-atom).curvature, u.Curvature.UNKNOWN)

        with self.assertRaises(Exception) as cm:
            normInf([[1,2],[3,4]])
        self.assertEqual(str(cm.exception), 
            "The argument '[[1, 2], [3, 4]]' to normInf must resolve to a vector.")

    # Test the norm1 class.
    def test_norm1(self):
        exp = self.x+self.y
        atom = norm1(exp)
        self.assertEquals(atom.name(), "norm1(x + y)")
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX)
        self.assertEquals(norm1(atom).curvature, u.Curvature.UNKNOWN)
        self.assertEquals(norm1(-atom).curvature, u.Curvature.UNKNOWN)

        with self.assertRaises(Exception) as cm:
            norm1([[1,2],[3,4]])
        self.assertEqual(str(cm.exception), 
            "The argument '[[1, 2], [3, 4]]' to norm1 must resolve to a vector.")

    # Test the norm2 class.
    def test_norm2(self):
        exp = self.x+self.y
        atom = norm2(exp)
        self.assertEquals(atom.name(), "norm2(x + y)")
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX)
        self.assertEquals(norm2(atom).curvature, u.Curvature.UNKNOWN)
        self.assertEquals(norm2(-atom).curvature, u.Curvature.UNKNOWN)

        with self.assertRaises(Exception) as cm:
            norm2([[1,2],[3,4]])
        self.assertEqual(str(cm.exception), 
            "The argument '[[1, 2], [3, 4]]' to norm2 must resolve to a vector.")

    # Test the vstack class.
    def test_vstack(self):
        atom = vstack(self.x, self.y, self.x)
        self.assertEquals(atom.name(), "vstack(x, y, x)")
        self.assertEquals(atom.size, (6,1))

        self.assertEqual(atom[0,0].name(), "x[0,0]")
        self.assertEqual(atom[2,0].name(), "y[0,0]")
        self.assertEqual(atom[5,0].name(), "x[1,0]")

        atom = vstack(self.A, self.C, self.B)
        self.assertEquals(atom.name(), "vstack(A, C, B)")
        self.assertEquals(atom.size, (7,2))

        self.assertEqual(atom[0,1].name(), "A[0,1]")
        self.assertEqual(atom[3,0].name(), "C[1,0]")
        self.assertEqual(atom[6,1].name(), "B[1,1]")

        gen = (xi for xi in self.x)
        atom = vstack(*gen)
        self.assertEqual(atom[1,0].name(), "x[1,0]")

        with self.assertRaises(Exception) as cm:
            vstack(self.C, 1)
        self.assertEqual(str(cm.exception), 
            "All arguments to vstack must have the same number of columns.")

        with self.assertRaises(Exception) as cm:
            vstack()
        self.assertEqual(str(cm.exception), 
            "No arguments given to 'vstack'.")