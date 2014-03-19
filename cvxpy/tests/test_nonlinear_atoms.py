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
import cvxpy.atoms.elementwise.log as cvxlog
from base_test import BaseTest
import cvxopt.solvers
import cvxopt
import unittest
import math
import numpy as np

class TestNonlinearAtoms(BaseTest):
    """ Unit tests for the nonlinear atoms module. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    # def test_log(self):
    #     """ Test that minimize -sum(log(x)) s.t. x <= 1 yields 0.

    #         Rewritten by hand.

    #         neg_log_func implements

    #             t1 - log(t2) <= 0

    #         Implemented as

    #             minimize [-1,-1,0,0] * [t1; t2]
    #                 t1 - log(t2) <= 0
    #                 [0 0 -1 0;
    #                  0 0 0 -1] * [t1; t2] <= [-1; -1]
    #     """
    #     F = cvxlog.neg_log_func(2)
    #     h = cvxopt.matrix([1.,1.])
    #     G = cvxopt.spmatrix([1.,1.], [0,1], [2,3], (2,4), tc='d')
    #     sol = cvxopt.solvers.cpl(cvxopt.matrix([-1.0,-1.0,0,0]), F, G, h)

    #     self.assertEqual(sol['status'], 'optimal')
    #     self.assertAlmostEqual(sol['x'][0], 0.)
    #     self.assertAlmostEqual(sol['x'][1], 0.)
    #     self.assertAlmostEqual(sol['x'][2], 1.)
    #     self.assertAlmostEqual(sol['x'][3], 1.)
    #     self.assertAlmostEqual(sol['primal objective'], 0.0)

    def test_log_problem(self):
        # Log in objective.
        obj = Maximize(sum(log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])

        # Log in constraint.
        obj = Minimize(sum(self.x))
        constr = [log(self.x) >= 0, self.x <= [1,1]]
        p = Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1,1])

        # Index into log.
        obj = Maximize(log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = Problem(obj,constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1)

    def test_entr(self):
        """Test the entr atom.
        """
        self.assertEqual(entr(0).value, 0)
        assert np.isneginf(entr(-1).value)

    # def test_kl_div(self):
    #     """Test the kl_div atom.
    #     """
    #     self.assertEqual(kl_div(0, 0).value, 0)
    #     self.assertEqual(kl_div(1, 0).value, np.inf)
    #     self.assertEqual(kl_div(0, 1).value, np.inf)
    #     self.assertEqual(kl_div(-1, -1).value, np.inf)

