"""
Copyright 2013 Steven Diamond, Eric Chu

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

from cvxpy import *
from cvxpy.expressions.variables import BoolVar, IntVar
from cvxopt import matrix
import numpy as np
from base_test import BaseTest
import unittest

class TestMIPVariable(BaseTest):
    """ Unit tests for the expressions/shape module. """
    def setUp(self):
        self.x_bool = BoolVar()
        self.y_int = IntVar()
        self.A_bool = BoolVar(3, 2)
        self.B_int = IntVar(2, 3)

    def test_mip_print(self):
        """Test to string methods for Bool/Int vars.
        """
        self.assertEqual(repr(self.x_bool), "BoolVar(1, 1)")
        self.assertEqual(repr(self.B_int), "IntVar(2, 3)")

    def test_bool_prob(self):
        # BoolVar in objective.
        obj = Minimize(square(self.x_bool - 0.2))
        p = Problem(obj,[])
        result = p.solve()
        self.assertAlmostEqual(result, 0.04)

        self.assertAlmostEqual(self.x_bool.value, 0)

        # BoolVar in constraint.
        t = Variable()
        obj = Minimize(t)
        p = Problem(obj,[square(self.x_bool) <= t])
        result = p.solve()
        self.assertAlmostEqual(result, 0)

        self.assertAlmostEqual(self.x_bool.value, 0)

        # Matrix BoolVar in objective.
        C = matrix([[0, 1, 0], [1, 1, 1]])
        obj = Minimize(sum_squares(self.A_bool - C))
        p = Problem(obj,[])
        result = p.solve()
        self.assertAlmostEqual(result, 0)

        self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)

        # Matrix BoolVar in constraint.
        t = Variable()
        obj = Minimize(t)
        p = Problem(obj, [sum_squares(self.A_bool - C) <= t])
        result = p.solve()
        self.assertAlmostEqual(result, 0)

        self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)

    def test_int_prob(self):
        # IntVar in objective.
        obj = Minimize(square(self.y_int - 0.2))
        p = Problem(obj,[])
        result = p.solve()
        self.assertAlmostEqual(result, 0.04)

        self.assertAlmostEqual(self.y_int.value, 0)
