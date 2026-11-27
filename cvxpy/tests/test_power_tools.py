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

from fractions import Fraction

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities import power_tools as pt


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
            log_solve_args = {'solver': 'CLARABEL'}
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


def test_power_tools_edge_branches():
    t = cp.Variable()
    x = cp.Variable()
    constraints = pt.gm_constrs(t, [x], (Fraction(1),))
    assert len(constraints) == 1
    assert constraints[0].args[0] is t
    assert constraints[0].args[1] is x

    assert pt.pow_high(2, approx=False) == (2, (Fraction(1, 2), Fraction(1, 2)))
    assert pt.pow_high(Fraction(3, 2))[0] == Fraction(3, 2)
    p, weights = pt.pow_neg(-2, approx=False)
    assert p == -2
    np.testing.assert_allclose(weights, (2 / 3, 1 / 3))
    assert pt.pow_neg(-2)[0] == -2
    assert pt.is_dyad(2)
    assert not pt.is_dyad(Fraction(1, 3))
    assert pt.is_weight(np.array([0, 0, 1]))

    with pytest.raises(ValueError, match="nonnegative"):
        pt.fracify([-1, 2])
    with pytest.raises(ValueError, match="denominator"):
        pt.fracify([1, 2], max_denom=0)
    with pytest.raises(ValueError, match="reliably represent"):
        pt.fracify((Fraction(1, 3), Fraction(2, 3)), max_denom=2)

    w, w_dyad = pt.fracify(np.array([0.1, 0.9]), max_denom=16)
    assert pt.is_weight(w)
    assert pt.is_dyad_weight(w_dyad)
    assert pt.make_frac([0.2, 0.8], 4) == (Fraction(1, 4), Fraction(3, 4))
    assert pt.dyad_completion((Fraction(1, 3), Fraction(2, 3))) == (
        Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)
    )
    assert pt.approx_error([1, 1], [Fraction(1, 2), Fraction(1, 2)]) == 0
    assert pt.next_pow2(0) == 1
    assert pt.check_dyad((Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 2)))
    assert not pt.check_dyad((Fraction(2, 3), 1), (Fraction(2, 3), 1))
    assert pt.split((Fraction(1), 0)) == ()
    tree = pt.decompose((Fraction(1, 2), Fraction(1, 2)))
    assert pt.over_bound((Fraction(1, 2), Fraction(1, 2)), tree) == 0
    pretty = pt.prettydict(tree)
    assert "(1/2, 1/2)" in pretty
