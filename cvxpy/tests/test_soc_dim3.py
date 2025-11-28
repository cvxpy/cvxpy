"""
Copyright 2025, the CVXPY authors

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
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.cone2cone.soc_dim3 import (
    ChainTree,
    LeafTree,
    NonNegTree,
    SOCDim3,
    SplitTree,
    _decompose_soc_single,
)
from cvxpy.reductions.solution import Solution
from cvxpy.tests.solver_test_helpers import socp_1, socp_2

# Dimensions to test: covers all decomposition branches
# dim 2 (NonNeg), dim 3 (Leaf), dim 4 (Chain), dim 5+ (Split variations)
TEST_DIMS = [1, 2, 3, 4, 5, 9, 19]


# =============================================================================
# Helper Functions
# =============================================================================

def _solve_with_reduction(prob):
    """Apply SOCDim3 reduction, solve, and return inverted solution."""
    reduction = SOCDim3()
    new_prob, inv_data = reduction.apply(prob)

    # Verify all SOC cones are dim-3
    for c in new_prob.constraints:
        if isinstance(c, SOC):
            for dim in c.cone_sizes():
                assert dim == 3, f"Expected dim 3, got {dim}"

    new_prob.solve(solver=cp.CLARABEL)

    sol = Solution(
        new_prob.status, new_prob.value,
        {v.id: v.value for v in new_prob.variables()},
        {c.id: c.dual_value for c in new_prob.constraints},
        {}
    )
    return reduction.invert(sol, inv_data), new_prob


def _flatten_dual(dual_value):
    """Flatten a dual value to 1D array for comparison."""
    if dual_value is None:
        return None
    if isinstance(dual_value, list) and len(dual_value) == 2:
        # SOC format: [t_array, x_array]
        return np.concatenate([
            np.atleast_1d(dual_value[0]).flatten(),
            np.atleast_1d(dual_value[1]).flatten()
        ])
    return np.atleast_1d(dual_value).flatten()


def _check_solution_matches(prob, inv_sol, new_prob, atol=1e-4):
    """Check that inverted solution matches direct solve for all vars and duals."""
    # Check objective value
    assert np.abs(new_prob.value - prob.value) < atol, (
        f"Objective mismatch: {new_prob.value} vs {prob.value}"
    )

    # Check all primal variables
    for v in prob.variables():
        direct_val = v.value
        inverted_val = inv_sol.primal_vars.get(v.id)
        if direct_val is not None:
            assert inverted_val is not None, f"Missing primal for {v.name()}"
            assert np.allclose(direct_val, inverted_val, atol=atol), (
                f"Primal mismatch for {v.name()}"
            )

    # Check all constraint duals
    for c in prob.constraints:
        direct_dual = c.dual_value
        inverted_dual = inv_sol.dual_vars.get(c.id)
        if direct_dual is not None:
            assert inverted_dual is not None, f"Missing dual for constraint {c.id}"
            direct_flat = _flatten_dual(direct_dual)
            inverted_flat = _flatten_dual(inverted_dual)
            assert np.allclose(direct_flat, inverted_flat, atol=atol), (
                f"Dual mismatch for constraint {c.id}"
            )


# =============================================================================
# Parameterized Tests
# =============================================================================

class TestSOCDim3Properties:
    """Parameterized tests for SOCDim3 reduction."""

    @pytest.mark.parametrize("x_size", TEST_DIMS)
    def test_decomposition_produces_only_dim3_cones(self, x_size):
        """All decomposed SOC cones are dimension 3."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(x_size)

        cones, nonneg_constrs = [], []
        _decompose_soc_single(t, x, cones, nonneg_constrs)

        for cone in cones:
            assert cone.cone_sizes() == [3], f"dim={x_size+1}: got {cone.cone_sizes()}"

    @pytest.mark.parametrize("x_size", TEST_DIMS)
    def test_solving_primal_and_dual(self, x_size):
        """Decomposed problem gives same primal and dual as direct solve."""
        x = cp.Variable(x_size)
        t = cp.Variable(nonneg=True)
        soc = SOC(t, x)
        x_val = np.arange(1, x_size + 1, dtype=float)
        prob = cp.Problem(cp.Minimize(t), [soc, x == x_val])

        # Direct solve
        prob.solve(solver=cp.CLARABEL)

        # Solve with decomposition and check everything matches
        inv_sol, new_prob = _solve_with_reduction(prob)
        _check_solution_matches(prob, inv_sol, new_prob)


# =============================================================================
# Decomposition Structure Tests
# =============================================================================

class TestSOCDim3Decomposition:
    """Test the SOC decomposition algorithm structure."""

    def test_dim2_to_nonneg(self):
        """Dimension 2 (|x| <= t) converts to NonNeg constraints."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(1)

        cones, nonneg_constrs = [], []
        tree = _decompose_soc_single(t, x, cones, nonneg_constrs)

        assert len(cones) == 0
        assert len(nonneg_constrs) == 2
        assert isinstance(tree, NonNegTree)
        assert tree.original_dim == 2

    def test_dim3_is_leaf(self):
        """Dimension 3 passes through as LeafTree."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(2)

        cones, nonneg_constrs = [], []
        tree = _decompose_soc_single(t, x, cones, nonneg_constrs)

        assert len(cones) == 1
        assert cones[0].cone_sizes() == [3]
        assert isinstance(tree, LeafTree)

    def test_dim4_is_chain(self):
        """Dimension 4 uses ChainTree structure."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(3)

        cones, nonneg_constrs = [], []
        tree = _decompose_soc_single(t, x, cones, nonneg_constrs)

        assert len(cones) == 2
        assert all(c.cone_sizes() == [3] for c in cones)
        assert isinstance(tree, ChainTree)

    def test_dim5_is_split(self):
        """Dimension 5+ uses SplitTree structure."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(4)

        cones, nonneg_constrs = [], []
        tree = _decompose_soc_single(t, x, cones, nonneg_constrs)

        assert len(cones) == 3
        assert all(c.cone_sizes() == [3] for c in cones)
        assert isinstance(tree, SplitTree)


# =============================================================================
# Standard SOCP Test Cases
# =============================================================================

class TestSOCDim3StandardProblems:
    """Test standard SOCP problems from solver test helpers."""

    def test_socp_1(self):
        """Standard SOCP test case 1 (dim-4 SOC)."""
        sth = socp_1()
        sth.solve(cp.CLARABEL)
        inv_sol, new_prob = _solve_with_reduction(sth.prob)
        _check_solution_matches(sth.prob, inv_sol, new_prob)

    def test_socp_2(self):
        """Standard SOCP test case 2 (dim-2 SOC)."""
        sth = socp_2()
        sth.solve(cp.CLARABEL)
        inv_sol, new_prob = _solve_with_reduction(sth.prob)
        _check_solution_matches(sth.prob, inv_sol, new_prob)


# =============================================================================
# Edge Cases
# =============================================================================

class TestSOCDim3EdgeCases:
    """Test edge cases for SOCDim3 reduction."""

    def test_multi_cone_soc(self):
        """SOC constraint with multiple elementwise cones (axis=0)."""
        n_cones, x_size = 3, 5
        t = cp.Variable(n_cones, nonneg=True)
        X = cp.Variable((x_size, n_cones))
        soc = SOC(t, X, axis=0)

        X_val = np.arange(1, x_size * n_cones + 1, dtype=float).reshape(x_size, n_cones)
        prob = cp.Problem(cp.Minimize(cp.sum(t)), [soc, X == X_val])

        prob.solve(solver=cp.CLARABEL)
        inv_sol, new_prob = _solve_with_reduction(prob)
        _check_solution_matches(prob, inv_sol, new_prob)

    def test_axis1_soc(self):
        """SOC constraint with axis=1."""
        n_cones, x_size = 2, 4
        t = cp.Variable(n_cones, nonneg=True)
        X = cp.Variable((n_cones, x_size))
        soc = SOC(t, X, axis=1)

        X_val = np.arange(1, n_cones * x_size + 1, dtype=float).reshape(n_cones, x_size)
        prob = cp.Problem(cp.Minimize(cp.sum(t)), [soc, X == X_val])

        prob.solve(solver=cp.CLARABEL)
        inv_sol, new_prob = _solve_with_reduction(prob)
        _check_solution_matches(prob, inv_sol, new_prob)

    def test_infeasible_propagates(self):
        """INFEASIBLE status propagates through decomposition."""
        x = cp.Variable(5)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.norm(x, 2) <= 1, cp.sum(x) >= 10]
        )

        prob.solve(solver=cp.CLARABEL)
        direct_status = prob.status

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        assert direct_status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]
        assert new_prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]

    def test_unbounded_propagates(self):
        """UNBOUNDED status propagates through decomposition."""
        x = cp.Variable(5)
        t = cp.Variable()
        prob = cp.Problem(cp.Minimize(-t), [cp.norm(x, 2) <= t])

        prob.solve(solver=cp.CLARABEL)
        direct_status = prob.status

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        assert direct_status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]
        assert new_prob.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]

    def test_multiple_soc_different_dims(self):
        """Multiple SOC constraints of different dimensions."""
        x = cp.Variable(5)
        y = cp.Variable(3)
        z = cp.Variable(10)
        t1, t2, t3 = cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True)
        soc1, soc2, soc3 = SOC(t1, x), SOC(t2, y), SOC(t3, z)

        prob = cp.Problem(
            cp.Minimize(cp.sum(x) + cp.sum(y) + cp.sum(z)),
            [soc1, soc2, soc3, t1 <= 1, t2 <= 2, t3 <= 3]
        )

        prob.solve(solver=cp.CLARABEL)
        inv_sol, new_prob = _solve_with_reduction(prob)
        _check_solution_matches(prob, inv_sol, new_prob)
