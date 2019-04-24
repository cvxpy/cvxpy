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
import unittest


class TestSolvers(BaseTest):
    """ Unit tests for solver specific behavior. """

    def setUp(self):
        self.a = cvx.Variable(name='a')
        self.b = cvx.Variable(name='b')
        self.c = cvx.Variable(name='c')

        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(3, name='y')
        self.z = cvx.Variable(2, name='z')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_ecos_options(self):
        """Test that all the ECOS solver options work.
        """
        # Test ecos
        # feastol, abstol, reltol, feastol_inacc,
        # abstol_inacc, and reltol_inacc for tolerance values
        # max_iters for the maximum number of iterations,
        EPS = 1e-4
        prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
        for i in range(2):
            prob.solve(solver=cvx.ECOS, feastol=EPS, abstol=EPS, reltol=EPS,
                       feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
                       max_iters=20, verbose=True, warm_start=True)
        self.assertAlmostEqual(prob.value, 1.0)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_ecos_bb_options(self):
        """Test that all the ECOS BB solver options work.
        """
        # 'mi_maxiter'
        # maximum number of branch and bound iterations (default: 1000)
        # 'mi_abs_eps'
        # absolute tolerance between upper and lower bounds (default: 1e-6)
        # 'mi_rel_eps'
        prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0),
                           [self.x == cvx.Variable(2, boolean=True)])
        for i in range(2):
            prob.solve(solver=cvx.ECOS_BB, mi_max_iters=100, mi_abs_eps=1e-6,
                       mi_rel_eps=1e-5, verbose=True, warm_start=True)
        self.assertAlmostEqual(prob.value, 1.0)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_scs_options(self):
        """Test that all the SCS solver options work.
        """
        # Test SCS
        # MAX_ITERS, EPS, ALPHA, UNDET_TOL, VERBOSE, and NORMALIZE.
        # If opts is missing, then the algorithm uses default settings.
        # USE_INDIRECT = True
        EPS = 1e-4
        prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
        for i in range(2):
            prob.solve(solver=cvx.SCS, max_iters=50, eps=EPS, alpha=EPS,
                       verbose=True, normalize=True, use_indirect=False)
        self.assertAlmostEqual(prob.value, 1.0, places=2)
        self.assertItemsAlmostEqual(self.x.value, [0, 0], places=2)

    def test_cvxopt_options(self):
        """Test that all the CVXOPT solver options work.
        """
        # TODO race condition when changing these values.
        # 'maxiters'
        # maximum number of iterations (default: 100).
        # 'abstol'
        # absolute accuracy (default: 1e-7).
        # 'reltol'
        # relative accuracy (default: 1e-6).
        # 'feastol'
        # tolerance for feasibility conditions (default: 1e-7).
        # 'refinement'
        # number of iterative refinement steps when solving KKT equations
        # (default: 0 if the problem has no second-order cone
        #  or matrix inequality constraints; 1 otherwise).
        if cvx.CVXOPT in cvx.installed_solvers():
            EPS = 1e-7
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
            for i in range(2):
                prob.solve(solver=cvx.CVXOPT, feastol=EPS, abstol=EPS, reltol=EPS,
                           max_iters=20, verbose=True, kktsolver="chol",
                           refinement=2, warm_start=True)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_cvxopt_glpk(self):
        """Test a basic LP with GLPK.
        """
        # Either the problem is solved or GLPK is not installed.
        if cvx.GLPK in cvx.installed_solvers():
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
            prob.solve(solver=cvx.GLPK)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GLPK)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.GLPK)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.GLPK)

    def test_cvxopt_glpk_mi(self):
        """Test a basic MILP with GLPK.
        """
        # Either the problem is solved or GLPK is not installed.
        if cvx.GLPK_MI in cvx.installed_solvers():
            bool_var = cvx.Variable(boolean=True)
            int_var = cvx.Variable(integer=True)
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0),
                               [self.x == bool_var, bool_var == 0])
            prob.solve(solver=cvx.GLPK_MI, verbose=True)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertAlmostEqual(bool_var.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= int_var,
                           self.x[0] + 2 * self.x[1] <= 3*bool_var,
                           self.x[0] >= 0,
                           self.x[1] >= 0,
                           int_var == 3*bool_var,
                           int_var == 3]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GLPK_MI, verbose=True)
            self.assertAlmostEqual(prob.value, -9)
            self.assertAlmostEqual(int_var.value, 3)
            self.assertAlmostEqual(bool_var.value, 1)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.GLPK_MI)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.GLPK_MI)

    def test_cplex(self):
        """Test a basic LP with CPLEX.
        """
        if cvx.CPLEX in cvx.installed_solvers():
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1), [self.x == 0])
            prob.solve(solver=cvx.CPLEX)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CPLEX)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # CPLEX's default lower bound for a decision variable is zero
            # This quick test ensures that the cvxpy interface for CPLEX does *not* have that bound
            objective = cvx.Minimize(self.x[0])
            constraints = [self.x[0] >= -100, self.x[0] <= -10, self.x[1] == 1]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CPLEX)
            self.assertItemsAlmostEqual(self.x.value, [-100, 1])

            # Boolean and integer version.
            bool_var = cvx.Variable(boolean=True)
            int_var = cvx.Variable(integer=True)
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)),
                               [self.x == bool_var, bool_var == 0])
            prob.solve(solver=cvx.CPLEX)
            self.assertAlmostEqual(prob.value, 0)
            self.assertAlmostEqual(bool_var.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= int_var,
                           self.x[0] + 2 * self.x[1] <= 3*bool_var,
                           self.x[0] >= 0,
                           self.x[1] >= 0,
                           int_var == 3*bool_var,
                           int_var == 3]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CPLEX)
            self.assertAlmostEqual(prob.value, -9)
            self.assertAlmostEqual(int_var.value, 3)
            self.assertAlmostEqual(bool_var.value, 1)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.CPLEX)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.CPLEX)

    def test_cplex_socp(self):
        """Test a basic SOCP with CPLEX.
        """
        if cvx.CPLEX in cvx.installed_solvers():
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 2) + 1.0), [self.x == 0])
            prob.solve(solver=cvx.CPLEX)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           (self.x[0] + 2 * self.x[1])**2 <= 9,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CPLEX)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # CPLEX's default lower bound for a decision variable is zero
            # This quick test ensures that the cvxpy interface for CPLEX does *not* have that bound
            objective = cvx.Minimize(self.x[0])
            constraints = [self.x[0] >= -100, self.x[0] <= -10, self.x[1] == 1]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CPLEX)
            self.assertItemsAlmostEqual(self.x.value, [-100, 1])

            # Boolean and integer version.
            bool_var = cvx.Variable(boolean=True)
            int_var = cvx.Variable(integer=True)
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 2)),
                               [self.x == bool_var, bool_var == 0])
            prob.solve(solver=cvx.CPLEX)
            self.assertAlmostEqual(prob.value, 0)
            self.assertAlmostEqual(bool_var.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= int_var,
                           (self.x[0] + 2 * self.x[1])**2 <= 9*bool_var,
                           self.x[0] >= 0,
                           self.x[1] >= 0,
                           int_var == 3*bool_var,
                           int_var == 3]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CPLEX)
            self.assertAlmostEqual(prob.value, -9)
            self.assertAlmostEqual(int_var.value, 3)
            self.assertAlmostEqual(bool_var.value, 1)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.CPLEX)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.CPLEX)

    def test_cplex_dual(self):
        """Make sure CPLEX's dual result matches other solvers
        """
        if cvx.CPLEX in cvx.installed_solvers():
            constraints = [self.x == 0]
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)))
            prob.solve(solver=cvx.CPLEX)
            duals_cplex = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_cplex, duals_ecos)

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CPLEX)
            duals_cplex = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_cplex, duals_ecos)

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.CPLEX)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.CPLEX)

    def test_cplex_warm_start(self):
        """Make sure that warm starting CPLEX behaves as expected
           Note: This only checks output, not whether or not CPLEX is warm starting internally
        """
        if cvx.CPLEX in cvx.installed_solvers():
            import numpy as np

            A = cvx.Parameter((2, 2))
            b = cvx.Parameter(2)
            h = cvx.Parameter(2)
            c = cvx.Parameter(2)

            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            objective = cvx.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
            constraints = [self.x[0] <= h[0],
                           self.x[1] <= h[1],
                           A * self.x == b]
            prob = cvx.Problem(objective, constraints)
            result = prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

            # Change A and b from the original values
            A.value = np.array([[0, 0], [0, 1]])   # <----- Changed
            b.value = np.array([0, 1])              # <----- Changed
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            # Without setting update_eq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])

            # Change h from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])              # <----- Changed
            c.value = np.array([1, 1])

            # Without setting update_ineq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # Change c from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])              # <----- Changed

            # Without setting update_objective = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.CPLEX)

    def test_cplex_params(self):
        if cvx.CPLEX in cvx.installed_solvers():
            import numpy.random as rnd

            n = 10
            m = 4
            A = rnd.randn(m, n)
            x = rnd.randn(n)
            y = A.dot(x)

            # Solve a simple basis pursuit problem for testing purposes.
            z = cvx.Variable(n)
            objective = cvx.Minimize(cvx.norm1(z))
            constraints = [A * z == y]
            problem = cvx.Problem(objective, constraints)

            invalid_cplex_params = {
                "bogus": "foo"
            }
            with self.assertRaises(ValueError):
                problem.solve(solver=cvx.CPLEX,
                              cplex_params=invalid_cplex_params)

            with self.assertRaises(ValueError):
                problem.solve(solver=cvx.CPLEX, invalid_kwarg=None)

            cplex_params = {
                "advance": 0,  # int param
                "simplex.limits.iterations": 1000,  # long param
                "timelimit": 1000.0,  # double param
                "workdir": '"mydir"',  # string param
            }
            problem.solve(solver=cvx.CPLEX, cplex_params=cplex_params)

    def test_cvxopt_dual(self):
        """Make sure CVXOPT's dual result matches other solvers
        """
        if cvx.CVXOPT in cvx.installed_solvers():
            constraints = [self.x == 0]
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)))
            prob.solve(solver=cvx.CVXOPT)
            duals_cvxopt = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_cvxopt, duals_ecos)

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CVXOPT)
            duals_cvxopt = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_cvxopt, duals_ecos)
        else:
            pass

    def test_gurobi(self):
        """Test a basic LP with Gurobi.
        """
        if cvx.GUROBI in cvx.installed_solvers():
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
            prob.solve(solver=cvx.GUROBI)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GUROBI)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # Gurobi's default lower bound for a decision variable is zero
            # This quick test ensures that the cvxpy interface for GUROBI does *not* have that bound
            objective = cvx.Minimize(self.x[0])
            constraints = [self.x[0] >= -100, self.x[0] <= -10, self.x[1] == 1]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GUROBI)
            self.assertItemsAlmostEqual(self.x.value, [-100, 1])

            # Boolean and integer version.
            bool_var = cvx.Variable(boolean=True)
            int_var = cvx.Variable(integer=True)
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)),
                               [self.x == bool_var, bool_var == 0])
            prob.solve(solver=cvx.GUROBI)
            self.assertAlmostEqual(prob.value, 0)
            self.assertAlmostEqual(bool_var.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= int_var,
                           self.x[0] + 2 * self.x[1] <= 3*bool_var,
                           self.x[0] >= 0,
                           self.x[1] >= 0,
                           int_var == 3*bool_var,
                           int_var == 3]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GUROBI)
            self.assertAlmostEqual(prob.value, -9)
            self.assertAlmostEqual(int_var.value, 3)
            self.assertAlmostEqual(bool_var.value, 1)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.GUROBI)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.GUROBI)

    def test_gurobi_socp(self):
        """Test a basic SOCP with Gurobi.
        """
        if cvx.GUROBI in cvx.installed_solvers():
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 2) + 1.0), [self.x == 0])
            prob.solve(solver=cvx.GUROBI)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           (self.x[0] + 2 * self.x[1])**2 <= 9,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GUROBI)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # Gurobi's default lower bound for a decision variable is zero
            # This quick test ensures that the cvxpy interface for GUROBI does *not* have that bound
            objective = cvx.Minimize(self.x[0])
            constraints = [self.x[0] >= -100, self.x[0] <= -10, self.x[1] == 1]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GUROBI)
            self.assertItemsAlmostEqual(self.x.value, [-100, 1])

            # Boolean and integer version.
            bool_var = cvx.Variable(boolean=True)
            int_var = cvx.Variable(integer=True)
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 2)),
                               [self.x == bool_var, bool_var == 0])
            prob.solve(solver=cvx.GUROBI)
            self.assertAlmostEqual(prob.value, 0)
            self.assertAlmostEqual(bool_var.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= int_var,
                           (self.x[0] + 2 * self.x[1])**2 <= 9*bool_var,
                           self.x[0] >= 0,
                           self.x[1] >= 0,
                           int_var == 3*bool_var,
                           int_var == 3]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GUROBI)
            self.assertAlmostEqual(prob.value, -9)
            self.assertAlmostEqual(int_var.value, 3)
            self.assertAlmostEqual(bool_var.value, 1)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.GUROBI)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.GUROBI)

    def test_gurobi_dual(self):
        """Make sure Gurobi's dual result matches other solvers
        """
        if cvx.GUROBI in cvx.installed_solvers():
            constraints = [self.x == 0]
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)))
            prob.solve(solver=cvx.GUROBI)
            duals_gurobi = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_gurobi, duals_ecos)

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GUROBI)
            duals_gurobi = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_gurobi, duals_ecos)

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.GUROBI)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.GUROBI)

    # I copied (and modified) the LP, SOCP, and dual GUROBI tests for MOSEK
    def test_mosek(self):
        """Test a basic LP with Mosek.
        """
        if cvx.MOSEK in cvx.installed_solvers():
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
            prob.solve(solver=cvx.MOSEK)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.MOSEK)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            objective = cvx.Minimize(self.x[0])
            constraints = [self.x[0] >= -100, self.x[0] <= -10, self.x[1] == 1]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.MOSEK)
            self.assertItemsAlmostEqual(self.x.value, [-100, 1])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.MOSEK)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.MOSEK)

    def test_mosek_socp(self):
        """Test a basic SOCP with Mosek.
        """
        if cvx.MOSEK in cvx.installed_solvers():
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 2) + 1.0), [self.x == 0])
            prob.solve(solver=cvx.MOSEK)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           (self.x[0] + 2 * self.x[1])**2 <= 9,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.MOSEK)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            objective = cvx.Minimize(self.x[0])
            constraints = [self.x[0] >= -100, self.x[0] <= -10, self.x[1] == 1]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.MOSEK)
            self.assertItemsAlmostEqual(self.x.value, [-100, 1])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.MOSEK)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.MOSEK)

    def test_mosek_dual(self):
        """Make sure Mosek's dual result matches other solvers
        """
        if cvx.MOSEK in cvx.installed_solvers():
            constraints = [self.x == 0]
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)))
            prob.solve(solver=cvx.MOSEK)
            duals_mosek = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_mosek, duals_ecos)

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.MOSEK)
            duals_mosek = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_mosek, duals_ecos)
        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.MOSEK)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.MOSEK)

    def test_mosek_sdp(self):
        """Make sure Mosek's dual result matches other solvers
        """
        # TODO: should work with PSD (>>, <<).
        if cvx.MOSEK in cvx.installed_solvers():
            # Test optimality gap for equilibration.
            n = 3
            Art = np.random.randn(n, n)

            t = cvx.Variable()
            d = cvx.Variable(n)
            D = cvx.diag(d)
            constr = [Art*D*Art.T - np.eye(n) == cvx.Variable((n, n), PSD=True),
                      cvx.Variable((n, n), PSD=True) == t*np.eye(n) - Art*D*Art.T, d >= 0]
            prob = cvx.Problem(cvx.Minimize(t), constr)
            prob.solve(solver=cvx.MOSEK)
        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.MOSEK)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.MOSEK)

    def test_mosek_params(self):
        if cvx.MOSEK in cvx.installed_solvers():
            import numpy.random as rnd
            import mosek

            n = 10
            m = 4
            A = rnd.randn(m, n)
            x = rnd.randn(n)
            y = A.dot(x)

            # Solve a simple basis pursuit problem for testing purposes.
            z = cvx.Variable(n)
            objective = cvx.Minimize(cvx.norm1(z))
            constraints = [A * z == y]
            problem = cvx.Problem(objective, constraints)

            invalid_mosek_params = {
                "dparam.basis_tol_x": "1e-8"
            }
            with self.assertRaises(ValueError):
                problem.solve(solver=cvx.MOSEK, mosek_params=invalid_mosek_params)

            with self.assertRaises(ValueError):
                problem.solve(solver=cvx.MOSEK, invalid_kwarg=None)

            mosek_params = {
                mosek.dparam.basis_tol_x: 1e-8,
                "MSK_IPAR_INTPNT_MAX_ITERATIONS": 20
            }
            problem.solve(solver=cvx.MOSEK, mosek_params=mosek_params)

    def test_gurobi_warm_start(self):
        """Make sure that warm starting Gurobi behaves as expected
           Note: This only checks output, not whether or not Gurobi is warm starting internally
        """
        if cvx.GUROBI in cvx.installed_solvers():
            import numpy as np

            A = cvx.Parameter((2, 2))
            b = cvx.Parameter(2)
            h = cvx.Parameter(2)
            c = cvx.Parameter(2)

            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            objective = cvx.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
            constraints = [self.x[0] <= h[0],
                           self.x[1] <= h[1],
                           A * self.x == b]
            prob = cvx.Problem(objective, constraints)
            result = prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

            # Change A and b from the original values
            A.value = np.array([[0, 0], [0, 1]])   # <----- Changed
            b.value = np.array([0, 1])              # <----- Changed
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            # Without setting update_eq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])

            # Change h from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])              # <----- Changed
            c.value = np.array([1, 1])

            # Without setting update_ineq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # Change c from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])              # <----- Changed

            # Without setting update_objective = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.GUROBI)

    def test_xpress(self):
        """Test a basic LP with Xpress.
        """
        if cvx.XPRESS in cvx.installed_solvers():
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
            prob.solve(solver=cvx.XPRESS)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.XPRESS)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            objective = cvx.Minimize(self.x[0])
            constraints = [self.x[0] >= -100, self.x[0] <= -10, self.x[1] == 1]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.XPRESS)
            self.assertItemsAlmostEqual(self.x.value, [-100, 1])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.XPRESS)
                self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.XPRESS)

    def test_xpress_socp(self):
        """Test a basic SOCP with Xpress.
        """
        if cvx.XPRESS in cvx.installed_solvers():
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 2) + 1.0), [self.x == 0])
            prob.solve(solver=cvx.XPRESS)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           (self.x[0] + 2 * self.x[1])**2 <= 9,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.XPRESS)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            objective = cvx.Minimize(self.x[0])
            constraints = [self.x[0] >= -100, self.x[0] <= -10, self.x[1] == 1]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.XPRESS)
            self.assertItemsAlmostEqual(self.x.value, [-100, 1])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.XPRESS)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.XPRESS)

    def test_xpress_dual(self):
        """Make sure Xpress' dual result matches other solvers
        """
        if cvx.XPRESS in cvx.installed_solvers():
            constraints = [self.x == 0]
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)))
            prob.solve(solver=cvx.XPRESS)
            duals_xpress = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_xpress, duals_ecos)

            # Example from
            # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = cvx.Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.XPRESS)
            duals_xpress = [x.dual_value for x in constraints]
            prob.solve(solver=cvx.ECOS)
            duals_ecos = [x.dual_value for x in constraints]
            self.assertItemsAlmostEqual(duals_xpress, duals_ecos)

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.XPRESS)
                self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.XPRESS)

    def test_installed_solvers(self):
        """Test the list of installed solvers.
        """
        from cvxpy.reductions.solvers.defines import (SOLVER_MAP_CONIC, SOLVER_MAP_QP,
                                                      INSTALLED_SOLVERS)
        prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
        for solver in SOLVER_MAP_CONIC.keys():
            if solver in INSTALLED_SOLVERS:
                prob.solve(solver=solver)
                self.assertAlmostEqual(prob.value, 1.0)
                self.assertItemsAlmostEqual(self.x.value, [0, 0])
            else:
                with self.assertRaises(Exception) as cm:
                    prob.solve(solver=solver)
                self.assertEqual(str(cm.exception), "The solver %s is not installed." % solver)

        for solver in SOLVER_MAP_QP.keys():
            if solver in INSTALLED_SOLVERS:
                prob.solve(solver=solver)
                self.assertItemsAlmostEqual(self.x.value, [0, 0])
            else:
                with self.assertRaises(Exception) as cm:
                    prob.solve(solver=solver)
                self.assertEqual(str(cm.exception), "The solver %s is not installed." % solver)


if __name__ == "__main__":
    unittest.main()
