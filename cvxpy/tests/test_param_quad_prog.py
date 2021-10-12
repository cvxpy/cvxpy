"""
Copyright, the CVXPY authors

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
import numpy as np

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS
from cvxpy.tests.base_test import BaseTest


class TestParamQuadProg(BaseTest):

    def setUp(self) -> None:
        self.solvers = [x for x in QP_SOLVERS if x in INSTALLED_SOLVERS]

    # Overridden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places: int = 2) -> None:
        super(TestParamQuadProg, self).assertItemsAlmostEqual(a, b, places=places)

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places: int = 2) -> None:
        super(TestParamQuadProg, self).assertAlmostEqual(a, b, places=places)

    def test_param_data(self) -> None:
        for solver in self.solvers:
            np.random.seed(0)
            m = 30
            n = 20
            A = np.random.randn(m, n)
            b = np.random.randn(m)
            x = cp.Variable(n)
            gamma = cp.Parameter(nonneg=True)
            gamma_val = .5
            gamma_val_new = .1
            objective = cp.Minimize(gamma * cp.sum_squares(A @ x - b) + cp.norm(x, 1))
            constraints = [1 <= x, x <= 2]

            # Solve from scratch (directly new parameter)
            prob = cp.Problem(objective, constraints)
            self.assertTrue(prob.is_dpp())
            gamma.value = gamma_val_new
            data_scratch, _, _ = prob.get_problem_data(solver)
            prob.solve(solver=solver)
            x_scratch = np.copy(x.value)

            # Canonicalize problem with parameter values (solve once)
            prob = cp.Problem(objective, constraints)
            gamma.value = gamma_val
            data_param, _, _ = prob.get_problem_data(solver)
            prob.solve(solver=solver)

            # Get data with new parameter
            gamma.value = gamma_val_new
            data_param_new, _, _ = prob.get_problem_data(solver)
            prob.solve(solver=solver)
            x_gamma_new = np.copy(x.value)

            # Check if data match
            np.testing.assert_allclose(data_param_new['P'].todense(), data_scratch['P'].todense())

            # Check if solutions match
            np.testing.assert_allclose(x_gamma_new, x_scratch, rtol=1e-02, atol=1e-02)

    def test_qp_problem(self) -> None:
        for solver in self.solvers:
            m = 30
            n = 20
            A = np.random.randn(m, n)
            b = np.random.randn(m)
            x = cp.Variable(n)
            gamma = cp.Parameter(nonneg=True)
            gamma.value = .5
            objective = cp.Minimize(cp.sum_squares(A @ x - b) + gamma * cp.norm(x, 1))
            constraints = [0 <= x, x <= 1]

            # Solve from scratch
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=solver)
            x_full = np.copy(x.value)

            # Restore cached values
            solving_chain = problem._cache.solving_chain
            solver = problem._cache.solving_chain.solver
            inverse_data = problem._cache.inverse_data
            param_prog = problem._cache.param_prog

            # Solve parametric
            data, solver_inverse_data = solving_chain.solver.apply(param_prog)
            inverse_data = inverse_data + [solver_inverse_data]
            raw_solution = solver.solve_via_data(
                    data, warm_start=False, verbose=False, solver_opts={})
            problem.unpack_results(raw_solution, solving_chain, inverse_data)
            x_param = np.copy(x.value)

            np.testing.assert_allclose(x_param, x_full, rtol=1e-2, atol=1e-02)

        # TODO: Add derivatives and adjoint tests
