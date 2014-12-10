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
GNU General Public License for more detail

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy import *
import numpy as np
from cvxpy.tests.base_test import BaseTest

class TestNonOptimal(BaseTest):
    """ Unit tests for infeasible and unbounded problems. """

    def test_scalar_lp(self):
        """Test scalar LP problems.
        """
        x1 = Variable()
        x2 = Variable()
        obj = Minimize(-x1-x2)
        constraints = [2*x1 + x2 >= 1, x1 + 3*x2 >= 1, x1>= 0, x2>=0]
        p_unb = Problem(obj, constraints)
        p_inf = Problem(Minimize(x1), [0 <= x1, x1 <= -1])
        for solver in [ECOS, CVXOPT, SCS]:
            print(solver)
            p_unb.solve(solver=solver)
            self.assertEqual(p_unb.status, UNBOUNDED)
            p_inf.solve(solver=solver)
            self.assertEqual(p_inf.status, INFEASIBLE)

    def test_vector_lp(self):
        """Test vector LP problems.
        """
        # Infeasible and unbounded problems.
        x = Variable(5)
        p_inf = Problem(Minimize(sum_entries(x)),
                        [x >= 1,
                         x <= 0])
        p_unb = Problem(Minimize(sum_entries(x)), [x <= 1])
        for solver in [ECOS, CVXOPT, SCS]:
            print(solver)
            p_inf.solve(solver=solver)
            self.assertEqual(p_inf.status, INFEASIBLE)
            p_unb.solve(solver=solver)
            self.assertEqual(p_unb.status, UNBOUNDED)

    def test_inaccurate(self):
        """Test the optimal inaccurate status.
        """
        x = Variable(5)
        prob = Problem(Maximize(sum_entries(sqrt(x))), [x <= 0])
        result = prob.solve(solver=SCS)
        self.assertEquals(prob.status, OPTIMAL_INACCURATE)
        assert result is not None

    # def test_socp(self):
    #     """Test SOCP problems.
    #     """
    #     # Infeasible and unbounded problems.
    #     x = Variable(5)
    #     obj = Maximize(sum_entries(sqrt(x)))
    #     p_inf = Problem(obj,
    #                     [x >= 1,
    #                      x <= 0])
    #     p_unb = Problem(obj, [x >= 1])
    #     for solver in [ECOS, CVXOPT, SCS]:
    #         print(solver)
    #         p_inf.solve(solver=solver)
    #         self.assertEqual(p_inf.status, INFEASIBLE)
    #         p_unb.solve(solver=solver)
    #         self.assertEqual(p_unb.status, UNBOUNDED)

    # def test_scp(self):
    #     """Test SDP problems.
    #     """
    #     # Infeasible and unbounded problems.
    #     X = Variable(5, 5)
    #     obj = Maximize(lambda_min(X))
    #     p_inf = Problem(obj,
    #                     [X >= 1,
    #                      X <= 0])
    #     p_unb = Problem(obj)
    #     for solver in [CVXOPT, SCS]:
    #         print(solver)
    #         p_inf.solve(solver=solver)
    #         self.assertEqual(p_inf.status, INFEASIBLE)
    #         p_unb.solve(solver=solver)
    #         self.assertEqual(p_unb.status, UNBOUNDED)
