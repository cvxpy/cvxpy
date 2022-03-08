"""
Copyright 2022, the CVXPY developers

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
from cvxpy.tests.base_test import BaseTest


class TestGeoMean(BaseTest):

    def test_multi_step_dyad_completion(self) -> None:
        """
        Consider four market equilibrium problems.

        The budgets "b" in these problems are chosen so that canonicalization
        of geo_mean(u, b) hits a recursive code-path in power_tools.dyad_completion(...).

        The reference solution is computed by taking the log of the geo_mean objective,
        which has the effect of making the problem ExpCone representable.
        """
        if 'MOSEK' in cp.installed_solvers():
            log_solve_args = {'solver': 'MOSEK'}
        else:
            log_solve_args = {'solver': 'ECOS'}
        n_buyer = 5
        n_items = 7
        np.random.seed(0)
        V = 0.5 * (1 + np.random.rand(n_buyer, n_items))
        X = cp.Variable(shape=(n_buyer, n_items), nonneg=True)
        cons = [cp.sum(X, axis=0) <= 1]
        u = cp.sum(cp.multiply(V, X), axis=1)
        bs = np.array([
            [110, 14, 6, 77, 108],
            [15., 4., 8., 0., 9.],
            [14., 21., 217., 57., 6.],
            [3., 36., 77., 8., 8.]
        ])
        for i, b in enumerate(bs):
            log_objective = cp.Maximize(b @ cp.log(u))
            log_prob = cp.Problem(log_objective, cons)
            log_prob.solve(**log_solve_args)
            expect_X = X.value

            geo_objective = cp.Maximize(cp.geo_mean(u, b))
            geo_prob = cp.Problem(geo_objective, cons)
            geo_prob.solve()
            actual_X = X.value
            try:
                self.assertItemsAlmostEqual(actual_X, expect_X, places=3)
            except AssertionError as e:
                print(f'Failure at index {i} (when b={str(b)}).')
                log_prob.solve(**log_solve_args, verbose=True)
                print(X.value)
                geo_prob.solve(verbose=True)
                print(X.value)
                print('The valuation matrix was')
                print(V)
                raise e

    def test_3d_power_cone_approx(self):
        """
        Use
            geo_mean((x,y), (alpha, 1-alpha)) >= |z|
        as a reformulation of
            PowCone3D(x, y, z, alpha).

        Check validity of the reformulation by solving
        orthogonal projection problems.
        """
        if 'MOSEK' in cp.installed_solvers():
            proj_solve_args = {'solver': 'MOSEK'}
        else:
            proj_solve_args = {'solver': 'SCS', 'eps': 1e-10}
        min_numerator = 2
        denominator = 25
        x = cp.Variable(3)
        np.random.seed(0)
        y = 10 * np.random.rand(3)  # the third value doesn't matter
        for i, numerator in enumerate(range(min_numerator, denominator, 3)):
            alpha_float = numerator / denominator
            y[2] = (y[0] ** alpha_float) * (y[1] ** (1 - alpha_float)) + 0.05
            objective = cp.Minimize(cp.norm(y - x, 2))

            actual_constraints = [cp.constraints.PowCone3D(x[0], x[1], x[2],
                                                           [alpha_float])]
            actual_prob = cp.Problem(objective, actual_constraints)
            actual_prob.solve(**proj_solve_args)
            actual_x = x.value.copy()

            weights = np.array([alpha_float, 1 - alpha_float])
            approx_constraints = [cp.geo_mean(x[:2], weights) >= cp.abs(x[2])]
            approx_prob = cp.Problem(objective, approx_constraints)
            approx_prob.solve()
            approx_x = x.value.copy()
            try:
                self.assertItemsAlmostEqual(actual_x, approx_x, places=4)
            except AssertionError as e:
                print(f'Failure at index {i} (when alpha={alpha_float}).')
                raise e
