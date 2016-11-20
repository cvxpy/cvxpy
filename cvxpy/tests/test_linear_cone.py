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

import cvxpy
import cvxpy.settings as s
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable, NonNegative, Bool, Int
from cvxpy.expressions.constants import Parameter, Constant
import cvxpy.utilities as u
import numpy
import unittest
from cvxpy import Problem, Minimize, Maximize
from cvxpy.tests.base_test import BaseTest
from cvxpy.reductions.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.solver_interface.conic_solvers.ecos_conif import ECOS


class TestLinearCone(BaseTest):
    """ Unit tests for the domain module. """

    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    # Test scalar LP problems.
    def test_scalar_lp(self):
        # p = Problem(Minimize(3*self.a), [self.a >= 2])
        # self.assertTrue(ConeMatrixStuffing().accepts(p))
        # p_new = ConeMatrixStuffing().apply(p)

        p = Problem(Maximize(3*self.a - self.b),
                    [self.a <= 2, self.b == self.a, self.b <= 5])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        p_new, inv_data = ConeMatrixStuffing().apply(p)
        self.assertAlmostEqual(p_new.solve(), 4)
        sltn = ECOS().solve(p_new, False, False, {})
        self.assertAlmostEqual(sltn.opt_val, 4)

        # With a constant in the objective.
        p = Problem(Minimize(3*self.a - self.b + 100),
                    [self.a >= 2,
                     self.b + 5*self.c - 2 == self.a,
                     self.b <= 5 + self.c])
        self.assertTrue(ConeMatrixStuffing().accepts(p))

        p = Problem(Maximize(self.a), [self.a <= 2])
        self.assertTrue(ConeMatrixStuffing().accepts(p))

        # Unbounded problems.
        p = Problem(Maximize(self.a), [self.a >= 2])
        self.assertTrue(ConeMatrixStuffing().accepts(p))

        # Infeasible problems.
        p = Problem(Maximize(self.a), [self.a >= 2, self.a <= 1])
        self.assertTrue(ConeMatrixStuffing().accepts(p))

    # Test vector LP problems.
    def test_vector_lp(self):
        c = Constant(numpy.matrix([1, 2]).T).value
        p = Problem(Minimize(c.T*self.x), [self.x >= c])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        p_new = ConeMatrixStuffing().apply(p)

        A = Constant(numpy.matrix([[3, 5], [1, 2]]).T).value
        I = Constant([[1, 0], [0, 1]])
        p = Problem(Minimize(c.T*self.x + self.a),
                    [A*self.x >= [-1, 1],
                     4*I*self.z == self.x,
                     self.z >= [2, 2],
                     self.a >= 2])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        p_new = ConeMatrixStuffing().apply(p)

    # Test matrix LP problems.
    def test_matrix_lp(self):
        T = Constant(numpy.ones((2, 2))).value
        p = Problem(Minimize(1), [self.A == T])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        p_new = ConeMatrixStuffing().apply(p)

        T = Constant(numpy.ones((2, 3))*2).value
        c = Constant(numpy.matrix([3, 4]).T).value
        p = Problem(Minimize(1), [self.A >= T*self.C,
                                  self.A == self.B, self.C == T.T])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        p_new = ConeMatrixStuffing().apply(p)
