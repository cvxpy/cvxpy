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
import warnings

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.tests import base_test


class TestMixedCoordinates(base_test.BaseTest):

    def species(self, measured_total, measured_log_ratio):
        c = cp.MixedCoordinates(lin=[[1, 1]], log=[[-1, 1]])
        total, log_ratio = c.lin[0], c.log[0]
        prob = cp.Problem(
            cp.Minimize(cp.square(total - measured_total)
                        + cp.square(log_ratio - measured_log_ratio)),
            [total >= 1])
        return c, prob

    def test_two_species(self) -> None:
        """Fit a total and a log ratio of two concentrations.

        The model is a QP in the declared coordinates. Both measurements are
        attainable, so the recovered concentrations must match the closed form
        (a, b) = t / (1 + e^r) * (1, e^r).
        """
        c, prob = self.species(3.0, 0.7)
        self.assertTrue(prob.is_qp())
        self.assertIsNone(c.value)
        prob.solve()
        t, r = 3.0, 0.7
        self.assertItemsAlmostEqual(
            c.value, t / (1 + np.exp(r)) * np.array([1, np.exp(r)]), places=5)

    def test_active_constraint(self) -> None:
        """A total measured below its lower bound is clipped to the bound.

        The log ratio is fitted exactly and the dual of ``total >= 1`` is the
        slope 2 * (1 - 0.2) of the squared error at the bound.
        """
        c, prob = self.species(0.2, -1.0)
        prob.solve()
        self.assertAlmostEqual(np.sum(c.value), 1, places=5)
        self.assertAlmostEqual(np.log(c.value[1] / c.value[0]), -1, places=5)
        self.assertAlmostEqual(prob.constraints[0].dual_value, 1.6, places=4)

    def test_parameter_resolve(self) -> None:
        """Re-solve with a new Parameter value and recover the new x.

        The measured log ratio is a Parameter. After each solve the recovered
        x must reproduce that solve's total and log ratio.
        """
        c = cp.MixedCoordinates(lin=[[1, 1]], log=[[-1, 1]])
        r = cp.Parameter()
        prob = cp.Problem(cp.Minimize(cp.square(c.lin[0] - 3) + cp.square(c.log[0] - r)))
        self.assertTrue(prob.is_dpp())
        for r_value in [0.7, -2.0]:
            r.value = r_value
            prob.solve()
            self.assertAlmostEqual(np.sum(c.value), 3, places=5)
            self.assertAlmostEqual(np.log(c.value[1] / c.value[0]), r_value, places=5)

    def test_contingency_tables(self) -> None:
        """Smooth the log odds ratios of 2x2 tables while matching their margins.

        Each period's four margins have rank 3, so ``lin`` has redundant rows.
        The recovered tables must match the margins, and their log odds ratios
        must equal a direct smoothing fit of the noisy log odds ratios.
        """
        T = 12
        rng = np.random.default_rng(0)
        t = np.arange(T)
        row = (0.4 + 0.1 * np.cos(t / 4)) * (100 + 5 * t)
        n_t = 100 + 5 * t
        col = (0.5 + 0.1 * np.sin(t / 5)) * n_t
        noisy = 1 + 0.8 * np.sin(t / 3) + 0.3 * rng.standard_normal(T)

        # x stacks the tables (p11, p12, p21, p22) period by period.
        margins = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        odds = np.array([[1, -1, -1, 1]])
        c = cp.MixedCoordinates(lin=np.kron(np.eye(T), margins), log=np.kron(np.eye(T), odds))
        targets = np.column_stack([row, n_t - row, col, n_t - col]).ravel()
        lor = c.log
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(lor - noisy) + 5 * cp.sum_squares(cp.diff(lor, 2))),
            [c.lin == targets])
        prob.solve()

        p = c.value.reshape(T, 4)
        self.assertTrue(np.all(p > 0))
        self.assertItemsAlmostEqual(p[:, 0] + p[:, 1], row, places=4)
        self.assertItemsAlmostEqual(p[:, 0] + p[:, 2], col, places=4)
        s = cp.Variable(T)
        cp.Problem(cp.Minimize(cp.sum_squares(s - noisy)
                               + 5 * cp.sum_squares(cp.diff(s, 2)))).solve()
        self.assertItemsAlmostEqual(
            np.log(p[:, 0] * p[:, 3] / (p[:, 1] * p[:, 2])), s.value, places=4)

    def test_not_orthogonal(self) -> None:
        """A row of ``lin`` that sees ``log``'s direction is rejected at construction.

        ``x[0]`` is not orthogonal to the log ratio, so the error names both rows.
        """
        with pytest.raises(ValueError, match="Row 1 of `lin` and row 0 of `log`"):
            cp.MixedCoordinates(lin=[[1, 1], [1, 0]], log=[[-1, 1]])

    def test_shape_mismatch(self) -> None:
        """``lin`` and ``log`` must act on vectors of the same length."""
        with pytest.raises(ValueError, match="columns"):
            cp.MixedCoordinates(lin=[[1, 1]], log=[[1, -1, 0]])

    def test_boundary_not_attained(self) -> None:
        """An optimal total of 0 has no positive preimage.

        Pulling the total toward 0 puts the optimum on the boundary of the
        attainable totals, so ``value`` warns and returns None.
        """
        c = cp.MixedCoordinates(lin=[[1, 1]], log=[[-1, 1]])
        prob = cp.Problem(cp.Minimize(cp.square(c.lin[0]) + cp.square(c.log[0] - 0.3)))
        prob.solve()
        with pytest.warns(UserWarning, match="boundary"):
            self.assertIsNone(c.value)

    def test_underdetermined_is_deterministic(self) -> None:
        """With rank(lin) + rank(log) < n, x doesn't depend on the solver's z.

        One total and one log ratio leave x[2] free. Solving with two solvers
        must give the same x, which matches both measurements and satisfies
        log(x) - z0 in row(lin), the KL optimality condition.
        """
        c = cp.MixedCoordinates(lin=[[1, 1, 1]], log=[[-1, 1, 0]])
        prob = cp.Problem(cp.Minimize(cp.square(c.lin[0] - 6) + cp.square(c.log[0] - 1)))
        values = []
        for solver in [cp.CLARABEL, cp.SCS]:
            prob.solve(solver=solver)
            values.append(c.value)
        self.assertItemsAlmostEqual(values[0], values[1], places=3)
        x = values[0]
        self.assertAlmostEqual(np.sum(x), 6, places=4)
        self.assertAlmostEqual(np.log(x[1] / x[0]), 1, places=4)
        z0 = np.array([-0.5, 0.5, 0])
        residual = np.log(x) - z0
        self.assertItemsAlmostEqual(residual, np.full(3, residual.mean()), places=4)

    def test_lin_only_and_log_only(self) -> None:
        """Only one of ``lin`` and ``log`` may be given.

        With only ``log``, x = exp(z0). Accessing the missing one raises.
        """
        c = cp.MixedCoordinates(log=[[-1, 1]])
        cp.Problem(cp.Minimize(cp.square(c.log[0] - 2))).solve()
        self.assertAlmostEqual(np.log(c.value[1] / c.value[0]), 2, places=5)
        with pytest.raises(AttributeError):
            c.lin
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c = cp.MixedCoordinates(lin=[[1, 1]])
            cp.Problem(cp.Minimize(cp.square(c.lin[0] - 2))).solve()
            self.assertItemsAlmostEqual(c.value, [1, 1], places=4)
