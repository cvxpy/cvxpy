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

import cvxpy as cvx
import numpy as np
from cvxpy.tests.base_test import BaseTest


class TestMosek(BaseTest):
    """ Unit tests for solver specific behavior. """

    def test_mosek_socp(self):
        """
        Formulate the following SOCP with cvxpy
            min 3 * x[0] + 2 * x[1] + x[2]
            s.t. norm(x,2) <= y[0]
                 norm(x,2) <= y[1]
                 x[0] + x[1] + 3*x[2] >= 1.0
                 y <= 5
        and solve with MOSEK and ECOS. Compare MOSEK and ECOS primal and dual solutions.
        """
        if cvx.MOSEK in cvx.installed_solvers():
            x = cvx.Variable(shape=(3,))
            y = cvx.Variable(shape=(2,))
            constraints = [cvx.norm(x, 2) <= y[0],
                           cvx.norm(x, 2) <= y[1],
                           x[0] + x[1] + 3 * x[2] >= 1.0,
                           y <= 5]
            obj = cvx.Minimize(3 * x[0] + 2 * x[1] + x[2])
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.MOSEK)
            x_mosek = x.value.tolist()
            duals_mosek = [c.dual_value for c in constraints]
            prob.solve(solver=cvx.ECOS)
            x_ecos = x.value.tolist()
            duals_ecos = [c.dual_value for c in constraints]
            self.assertItemsAlmostEqual(x_mosek, x_ecos)
            self.assertEqual(len(duals_ecos), len(duals_mosek))
            for i in range(len(duals_mosek)):
                if isinstance(duals_mosek[i], float):
                    self.assertAlmostEqual(duals_mosek[i], duals_ecos[i], places=4)
                else:
                    self.assertItemsAlmostEqual(duals_mosek[i].tolist(),
                                                duals_ecos[i].tolist(),
                                                places=4)
        else:
            pass

    def test_mosek_sdp(self):
        """
        Solve "Example 8.3" from Convex Optimization by Boyd & Vandenberghe.

        Verify (1) optimal objective values, (2) that the dual variable to the PSD constraint
        belongs to the correct cone (i.e. the dual variable is itself PSD), and (3) that
        complementary slackness holds with the PSD primal variable and its dual variable.
        """
        if cvx.MOSEK in cvx.installed_solvers():
            # This is an example from Convex Optimization by B&V.
            # Example 8.3 (page 408 in the 19th printing).
            rho = cvx.Variable(shape=(4, 4))
            constraints = [0.6 <= rho[0, 1], rho[0, 1] <= 0.9,
                           0.8 <= rho[0, 2], rho[0, 2] <= 0.9,
                           0.5 <= rho[1, 3], rho[1, 3] <= 0.7,
                           -0.8 <= rho[2, 3], rho[2, 3] <= -0.4,
                           rho[0, 0] == 1, rho[1, 1] == 1, rho[2, 2] == 1, rho[3, 3] == 1,
                           rho >> 0]
            # First, minimize rho[0, 3]
            obj = cvx.Minimize(rho[0, 3])
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.MOSEK)
            self.assertItemsAlmostEqual(np.round(prob.value, 2), -0.39)
            Y = constraints[-1].dual_value
            eigs = np.linalg.eig(Y)
            self.assertTrue(np.all(eigs[0] >= 0))
            complementary_slackness = np.trace(np.dot(rho.value, Y))
            self.assertAlmostEqual(complementary_slackness, 0.0)

            # Then, maximize rho[0, 3]
            obj = cvx.Maximize(rho[0, 3])
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.MOSEK)
            self.assertItemsAlmostEqual(np.round(prob.value, 2), 0.23)
            Y = constraints[-1].dual_value
            eigs = np.linalg.eig(Y)
            self.assertTrue(np.all(eigs[0] >= 0))
            complementary_slackness = np.trace(np.dot(rho.value, Y))
            self.assertAlmostEqual(complementary_slackness, 0.0)
        else:
            pass

    def test_mosek_exp(self):
        """
        Formulate the following exponential cone problem with cvxpy
            min   3 * x[0] + 2 * x[1] + x[2]
            s.t.  0.1 <= x[0] + x[1] + x[2] <= 1
                  x >= 0
                  x[0] >= x[1] * exp(x[2] / x[1])
        and solve with MOSEK and ECOS. Ensure that MOSEK and ECOS have the same
        primal and dual solutions.

        Note that the exponential cone constraint can be rewritten in terms of the
        relative entropy cone. The correspondence is as follows:
                x[0] >= x[1] * exp(x[2] / x[1])
            iff
                x[1] * log(x[1] / x[0]) + x[2] <= 0.
        """
        if cvx.MOSEK in cvx.installed_solvers():
            import mosek
            if hasattr(mosek.conetype, 'pexp'):
                # Formulate and solve the problem with CVXPY
                x = cvx.Variable(shape=(3, 1))
                constraints = [cvx.sum(x) <= 1.0, cvx.sum(x) >= 0.1, x >= 0.01,
                               cvx.kl_div(x[1], x[0]) + x[1] - x[0] + x[2] <= 0]
                obj = cvx.Minimize(3 * x[0] + 2 * x[1] + x[0])
                prob = cvx.Problem(obj, constraints)
                prob.solve(solver=cvx.MOSEK)
                val_mosek = prob.value
                x_mosek = x.value.flatten().tolist()
                duals_mosek = [c.dual_value.flatten().tolist() for c in constraints]
                prob.solve(solver=cvx.ECOS)
                val_ecos = prob.value
                x_ecos = x.value.flatten().tolist()
                duals_ecos = [c.dual_value.flatten().tolist() for c in constraints]

                # verify results
                self.assertAlmostEqual(val_mosek, val_ecos)
                self.assertItemsAlmostEqual(x_mosek, x_ecos, places=4)
                self.assertEqual(len(duals_ecos), len(duals_mosek))
                for i in range(len(duals_mosek)):
                    if isinstance(duals_mosek[i], float):
                        self.assertAlmostEqual(duals_mosek[i], duals_ecos[i], places=4)
                    else:
                        self.assertItemsAlmostEqual(duals_mosek[i], duals_ecos[i], places=4)
            else:
                pass

    def test_mosek_mi_socp(self):
        """
        Formulate the following mixed-integer SOCP with cvxpy
            min 3 * x[0] + 2 * x[1] + x[2] +  y[0] + 2 * y[1]
            s.t. norm(x,2) <= y[0]
                 norm(x,2) <= y[1]
                 x[0] + x[1] + 3*x[2] >= 0.1
                 y <= 5, y integer.
        and solve with MOSEK.
        """
        if cvx.MOSEK in cvx.installed_solvers():
            x = cvx.Variable(shape=(3,))
            y = cvx.Variable(shape=(2,), integer=True)
            constraints = [cvx.norm(x, 2) <= y[0],
                           cvx.norm(x, 2) <= y[1],
                           x[0] + x[1] + 3 * x[2] >= 0.1,
                           y <= 5]
            obj = cvx.Minimize(3 * x[0] + 2 * x[1] + x[2] + y[0] + 2 * y[1])
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.MOSEK)
            self.assertItemsAlmostEqual([1, 1], y.value.tolist())
        else:
            pass

    def test_mosek_LP_solution_selection(self):
        if cvx.MOSEK in cvx.installed_solvers():
            # Problem from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            x = cvx.Variable(shape=(2,))
            objective = cvx.Minimize(-4 * x[0] - 5 * x[1])
            constraints = [2 * x[0] + x[1] <= 3,
                           x[0] + 2 * x[1] <= 3,
                           x[0] >= 0, x[1] >= 0]
            prob = cvx.Problem(objective, constraints)

            # Default solution (interior point)
            prob.solve(solver=cvx.MOSEK)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(x.value, [1, 1], places=5)

            # Basic feasible solution
            prob.solve(solver=cvx.MOSEK, bfs=True)
            self.assertAlmostEqual(prob.value, -9, places=10)
            self.assertItemsAlmostEqual(x.value, [1, 1], places=10)
