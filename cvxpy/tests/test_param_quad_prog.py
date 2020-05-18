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
import math

import cvxpy as cp
from cvxpy.reductions.solvers.defines import QP_SOLVERS, INSTALLED_SOLVERS
from cvxpy.tests.base_test import BaseTest

import numpy as np


class TestParamQuadProg(BaseTest):

    def setUp(self):
        self.solvers = [x for x in QP_SOLVERS if x in INSTALLED_SOLVERS]

    # Overridden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places=2):
        super(TestParamQuadProg, self).assertItemsAlmostEqual(a, b, places=places)

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=2):
        super(TestParamQuadProg, self).assertAlmostEqual(a, b, places=places)

    def test_qp_problem(self):
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

            self.assertItemsAlmostEqual(x_param, x_full)


        # TODO: Add derivatives and adjoint tests
