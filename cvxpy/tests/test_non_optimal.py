"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestNonOptimal(BaseTest):
    """ Unit tests for infeasible and unbounded problems. """

    def test_scalar_lp(self):
        """Test scalar LP problems.
        """
        x1 = cp.Variable()
        x2 = cp.Variable()
        obj = cp.Minimize(-x1-x2)
        constraints = [2*x1 + x2 >= 1, x1 + 3*x2 >= 1, x1 >= 0, x2 >= 0]
        p_unb = cp.Problem(obj, constraints)
        p_inf = cp.Problem(cp.Minimize(x1), [0 <= x1, x1 <= -1])
        for solver in [cp.ECOS, cp.CVXOPT, cp.SCS]:
            if cp.CVXOPT in cp.installed_solvers():
                print(solver)
                p_unb.solve(solver=solver)
                self.assertEqual(p_unb.status, cp.UNBOUNDED)
                p_inf.solve(solver=solver)
                self.assertEqual(p_inf.status, cp.INFEASIBLE)

    def test_vector_lp(self):
        """Test vector LP problems.
        """
        # Infeasible and unbounded problems.
        x = cp.Variable(5)
        p_inf = cp.Problem(cp.Minimize(cp.sum(x)),
                           [x >= 1,
                           x <= 0])
        p_unb = cp.Problem(cp.Minimize(cp.sum(x)), [x <= 1])
        for solver in [cp.ECOS, cp.CVXOPT, cp.SCS]:
            if cp.CVXOPT in cp.installed_solvers():
                print(solver)
                p_inf.solve(solver=solver)
                self.assertEqual(p_inf.status, cp.INFEASIBLE)
                p_unb.solve(solver=solver)
                self.assertEqual(p_unb.status, cp.UNBOUNDED)

    # def test_inaccurate(self):
    #     """Test the optimal inaccurate status.
    #     """
    #     x = Variable(5)
    #     prob = Problem(Maximize(sum(sqrt(x))), [x <= 0])
    #     result = prob.solve(solver=SCS)
    #     self.assertEqual(prob.status, OPTIMAL_INACCURATE)
    #     assert result is not None

    # def test_socp(self):
    #     """Test SOCP problems.
    #     """
    #     # Infeasible and unbounded problems.
    #     x = Variable(5)
    #     obj = Maximize(sum(sqrt(x)))
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
    #     """Test PSD problems.
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
