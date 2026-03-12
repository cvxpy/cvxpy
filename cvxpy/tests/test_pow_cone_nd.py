"""
Copyright 2025 CVXPY developers

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
import pytest

import cvxpy as cp
from cvxpy.constraints.power import PowConeND
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS as SCS_SOLVER


def solve_prob(prob, solver):
    prob.solve(solver=solver, verbose=False)
    assert prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]


class TestPowConeND:
    """Unit tests for PowConeND and PowCone3D."""

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("solver", [cp.CLARABEL, cp.SCS])
    def test_pow_cone_nd_3d(self, axis, solver) -> None:
        """
        A variation of test_pcp_3.
        Clarabel natively supports ND power cones; SCS only supports 3D power
        cones so CVXPY applies a reduction.
        """
        # SCS does not natively support PowConeND, so the decomposition
        # to PowCone3D is exercised when solving with SCS.
        assert PowConeND not in SCS_SOLVER.SUPPORTED_CONSTRAINTS
        expect_x = np.array([0.06393515, 0.78320961, 2.30571048])
        x = cp.Variable(3, name='x')
        hypos = cp.Variable(2, name='hypos')
        objective = cp.Maximize(cp.sum(hypos) - x[0])
        W = cp.bmat([[x[0], x[2]],
                     [x[1], 1.0]])
        alpha = np.array([[0.2, 0.4],
                          [0.8, 0.6]])
        if axis == 1:
            W = W.T
            alpha = alpha.T

        constraints = [
            x[0] + x[1] + 0.5 * x[2] == 2,
            cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
        ]
        prob = cp.Problem(objective, constraints)
        solve_prob(prob, solver)
        np.testing.assert_allclose(x.value, expect_x, atol=1e-3)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("solver", [cp.CLARABEL, cp.SCS])
    def test_pow_cone_nd_3d_variable_swap(self, axis, solver) -> None:
        """
        A variation of test_pcp_3 with variable appearing in a different order.
        """
        expect_x = np.array([0.06393515, 2.30571048, 0.78320961])
        x = cp.Variable(3)
        hypos = cp.Variable(2)
        objective = cp.Maximize(cp.sum(hypos) - x[0])
        W = cp.bmat([[x[0], x[1]],
                     [x[2], 1.0]])
        alpha = np.array([[0.2, 0.4],
                          [0.8, 0.6]])
        if axis == 1:
            W = W.T
            alpha = alpha.T
        constraints = [
            x[0] + x[2] + 0.5 * x[1] == 2,
            cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
        ]
        prob = cp.Problem(objective, constraints)
        solve_prob(prob, solver)
        np.testing.assert_allclose(x.value, expect_x, atol=1e-3)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("solver", [cp.CLARABEL, cp.SCS])
    def test_pow_cone_nd(self, axis, solver) -> None:
        """Solving a PowConeND constraint with more than 3 dimensions."""
        expect_x = np.array([0, 0, 0, 2.28571379, 3.42857186])
        x = cp.Variable(5)
        hypos = cp.Variable(2)
        objective = cp.Maximize(cp.sum(hypos) - x[0])
        W = cp.bmat([[x[0], x[3]],
                     [x[1], x[4]],
                     [x[2], 1.0]])
        alpha = np.array([[0.2, 0.4],
                          [0.4, 0.3],
                          [0.4, 0.3]])
        if axis == 1:
            W = W.T
            alpha = alpha.T
        constraints = [
            x[0] + x[1] + 0.5 * x[2] + 0.5 * x[3] + 0.25 * x[4] == 2,
            cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
        ]
        prob = cp.Problem(objective, constraints)
        solve_prob(prob, solver)
        np.testing.assert_allclose(x.value, expect_x, atol=1e-3)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("solver", [cp.CLARABEL, cp.SCS])
    def test_pow_cone_nd_variable_swap(self, axis, solver) -> None:
        """
        A variation of test_pow_cone_nd with variable appearing in a different order.
        """
        expect_x = np.array([3.42857186, 0, 0, 2.28571379, 0])
        x = cp.Variable(5)
        hypos = cp.Variable(2)
        objective = cp.Maximize(cp.sum(hypos) - x[4])
        W = cp.bmat([[x[4], x[3]],
                     [x[1], x[0]],
                     [x[2], 1.0]])
        alpha = np.array([[0.2, 0.4],
                          [0.4, 0.3],
                          [0.4, 0.3]])
        if axis == 1:
            W = W.T
            alpha = alpha.T
        constraints = [
            x[4] + x[1] + 0.5 * x[2] + 0.5 * x[3] + 0.25 * x[0] == 2,
            cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
        ]
        prob = cp.Problem(objective, constraints)
        solve_prob(prob, solver)
        np.testing.assert_allclose(x.value, expect_x, atol=1e-3)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("solver", [cp.CLARABEL, cp.SCS])
    def test_pow_cone_nd_single_cone(self, axis, solver) -> None:
        """
        Solving a PowConeND constraint with only a single cone.
        Ensures no variables collapse to lower dimensions incorrectly.
        """
        x = cp.Variable(2)
        hypos = cp.Variable(1)
        objective = cp.Maximize(cp.sum(hypos) - x[0])
        W = cp.bmat([[x[0]], [x[1]]])
        alpha = np.array([[0.2], [0.8]])
        if axis == 1:
            W = W.T
            alpha = alpha.T
        constraints = [
            x[0] + x[1] == 2,
            cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
        ]
        prob = cp.Problem(objective, constraints)
        solve_prob(prob, solver)

    @pytest.mark.parametrize("solver", [cp.CLARABEL, cp.SCS])
    def test_3d_pow_cone_scalar_alpha(self, solver) -> None:
        """Test PowCone3D with scalar alpha."""
        x = cp.Variable(3)
        constraints = [cp.PowCone3D(x[0], x[1], x[2], 0.75)]
        prob = cp.Problem(cp.Minimize(cp.norm(x)), constraints)
        solve_prob(prob, solver)

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 100])
    @pytest.mark.parametrize("solver,atol", [(cp.CLARABEL, 1e-4), (cp.SCS, 1e-3)])
    def test_pow_cone_nd_balanced_tree_decomposition(self, n, solver, atol) -> None:
        """Test balanced tree decomposition for various n values."""
        W = cp.Variable(n, pos=True)
        z = cp.Variable()
        alpha = np.ones(n) / n
        con = cp.constraints.PowConeND(W, z, alpha)
        prob = cp.Problem(cp.Maximize(z), [con, W <= 2])
        solve_prob(prob, solver)
        np.testing.assert_allclose(z.value, 2.0, atol=atol)

    @pytest.mark.parametrize("solver,atol", [(cp.CLARABEL, 1e-4), (cp.SCS, 1e-3)])
    def test_pow_cone_nd_balanced_tree_nonuniform_alpha(self, solver, atol) -> None:
        """Test balanced tree decomposition with non-uniform alpha weights."""
        alpha = np.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])
        bounds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        expected = np.prod(bounds ** alpha)

        W = cp.Variable(8, pos=True)
        z = cp.Variable()
        con = cp.constraints.PowConeND(W, z, alpha)
        prob = cp.Problem(cp.Maximize(z), [con, W <= bounds])
        solve_prob(prob, solver)
        np.testing.assert_allclose(z.value, expected, atol=atol)

    @pytest.mark.parametrize("solver,atol", [(cp.CLARABEL, 1e-4), (cp.SCS, 1e-3)])
    def test_pow_cone_nd_balanced_tree_multiple_cones(self, solver, atol) -> None:
        """Test balanced tree decomposition with multiple cones (k > 1)."""
        n, k = 8, 3
        W = cp.Variable((n, k), pos=True)
        z = cp.Variable(k)
        alpha = np.ones((n, k)) / n
        con = cp.constraints.PowConeND(W, z, alpha, axis=0)
        prob = cp.Problem(cp.Maximize(cp.sum(z)), [con, W <= 2])
        solve_prob(prob, solver)
        np.testing.assert_allclose(z.value, [2.0, 2.0, 2.0], atol=atol)

    @pytest.mark.parametrize("solver,atol", [(cp.CLARABEL, 1e-4), (cp.SCS, 1e-3)])
    def test_pow_cone_nd_balanced_tree_multiple_cones_nonuniform(self, solver, atol) -> None:
        """Test balanced tree with non-uniform alpha and multiple cones (k > 1)."""
        n, k = 6, 2
        bounds = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ])
        alpha = np.array([
            [0.1, 0.3],
            [0.2, 0.2],
            [0.3, 0.1],
            [0.15, 0.15],
            [0.15, 0.1],
            [0.1, 0.15],
        ])
        expected = np.prod(bounds ** alpha, axis=0)

        W = cp.Variable((n, k), pos=True)
        z = cp.Variable(k)
        con = cp.constraints.PowConeND(W, z, alpha, axis=0)
        prob = cp.Problem(cp.Maximize(cp.sum(z)), [con, W <= bounds])
        solve_prob(prob, solver)
        np.testing.assert_allclose(z.value, expected, atol=atol)

    @pytest.mark.parametrize("n,rtol", [
        (2, 1e-3),   # n=2: no tree needed
        (3, 1e-2),   # n=3: depth 1
        (4, 1e-2),   # n=4: depth 2
        (8, 1e-2),   # n=8: depth 3
        (16, 5e-2),  # n=16: depth 4
    ])
    def test_pow_cone_nd_dual_variables(self, n, rtol) -> None:
        """Test dual variable recovery matches CLARABEL."""
        W = cp.Variable((n, 1), pos=True)
        z = cp.Variable(1)
        alpha = np.ones((n, 1)) / n
        con = cp.constraints.PowConeND(W, z, alpha, axis=0)
        prob = cp.Problem(cp.Maximize(z[0]), [con, W <= 2])

        solve_prob(prob, cp.CLARABEL)
        clarabel_w_dual = con.dual_value[0].flatten()
        clarabel_z_dual = con.dual_value[1].flatten()

        solve_prob(prob, cp.SCS)
        scs_w_dual = con.dual_value[0].flatten()
        scs_z_dual = con.dual_value[1].flatten()

        np.testing.assert_allclose(scs_w_dual, clarabel_w_dual, rtol=rtol)
        np.testing.assert_allclose(scs_z_dual, clarabel_z_dual, rtol=rtol)
