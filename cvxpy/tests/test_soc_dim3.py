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

# Dimensions to test: covers all decomposition branches
# dim 2 (NonNeg), dim 3 (Leaf), dim 4 (Chain), dim 5+ (Split variations)
TEST_DIMS = [1, 2, 3, 4, 5, 9, 19]


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

        cones = []
        nonneg_constrs = []
        _decompose_soc_single(t, x, cones, nonneg_constrs)

        for cone in cones:
            assert cone.cone_sizes() == [3], f"dim={x_size+1}: got {cone.cone_sizes()}"

    @pytest.mark.parametrize("x_size", TEST_DIMS)
    def test_solving_matches_direct(self, x_size):
        """Decomposed problem gives same objective as direct solve."""
        x = cp.Variable(x_size)
        c = np.ones(x_size)
        prob = cp.Problem(cp.Minimize(c @ x), [cp.norm(x, 2) <= 1])

        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        expected = -np.sqrt(x_size)
        assert np.abs(direct_val - expected) < 1e-4
        assert np.abs(new_prob.value - expected) < 1e-4

    @pytest.mark.parametrize("x_size", TEST_DIMS)
    def test_dual_reconstruction_accuracy(self, x_size):
        """Reconstructed duals match direct solve duals."""
        x = cp.Variable(x_size)
        t = cp.Variable(nonneg=True)
        soc = SOC(t, x)
        x_val = np.arange(1, x_size + 1, dtype=float)
        prob = cp.Problem(cp.Minimize(t), [soc, x == x_val])

        prob.solve(solver=cp.CLARABEL)
        direct_dual = soc.dual_value
        direct_flat = np.concatenate([
            np.atleast_1d(direct_dual[0]),
            np.atleast_1d(direct_dual[1]).flatten()
        ])

        reduction = SOCDim3()
        new_prob, inv_data = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        sol = Solution(
            new_prob.status, new_prob.value,
            {v.id: v.value for v in new_prob.variables()},
            {c.id: c.dual_value for c in new_prob.constraints},
            {}
        )
        inv_sol = reduction.invert(sol, inv_data)

        assert soc.id in inv_sol.dual_vars
        reconstructed = inv_sol.dual_vars[soc.id]
        assert reconstructed is not None
        assert len(reconstructed) == x_size + 1
        assert np.allclose(reconstructed, direct_flat, atol=1e-4)


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
        """Standard SOCP test case 1."""
        x = cp.Variable(shape=(3,))
        y = cp.Variable()
        soc = SOC(y, x)
        constraints = [soc, x[0] + x[1] + 3 * x[2] >= 1.0, y <= 5]
        prob = cp.Problem(cp.Minimize(3 * x[0] + 2 * x[1] + x[2]), constraints)

        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        expected = -13.548638904065102
        assert np.abs(direct_val - expected) < 1e-3
        assert np.abs(new_prob.value - expected) < 1e-3

    def test_socp_2(self):
        """Standard SOCP test case 2 (dim-2 SOC)."""
        x = cp.Variable(shape=(2,), name='x')
        expr = cp.reshape(x[0] + 2 * x[1], (1, 1), order='F')
        constraints = [
            2 * x[0] + x[1] <= 3,
            SOC(cp.Constant([3]), expr),
            x[0] >= 0, x[1] >= 0
        ]
        prob = cp.Problem(cp.Minimize(-4 * x[0] - 5 * x[1]), constraints)

        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        assert np.abs(direct_val - (-9.0)) < 1e-3
        assert np.abs(new_prob.value - (-9.0)) < 1e-3


# =============================================================================
# Edge Cases
# =============================================================================

class TestSOCDim3EdgeCases:
    """Test edge cases for SOCDim3 reduction."""

    def test_multi_cone_soc(self):
        """SOC constraint with multiple elementwise cones."""
        t = cp.Variable(3, nonneg=True)
        X = cp.Variable((5, 3))

        prob = cp.Problem(cp.Minimize(cp.sum(t)), [SOC(t, X, axis=0)])

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)

        for c in new_prob.constraints:
            if isinstance(c, SOC):
                for dim in c.cone_sizes():
                    assert dim == 3

    def test_axis1_soc(self):
        """SOC constraint with axis=1."""
        t = cp.Variable(2, nonneg=True)
        X = cp.Variable((2, 4))

        prob = cp.Problem(cp.Minimize(cp.sum(t)), [SOC(t, X, axis=1)])

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)

        for c in new_prob.constraints:
            if isinstance(c, SOC):
                for dim in c.cone_sizes():
                    assert dim == 3

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

        prob = cp.Problem(
            cp.Minimize(cp.sum(x) + cp.sum(y) + cp.sum(z)),
            [cp.norm(x, 2) <= 1, cp.norm(y, 2) <= 2, cp.norm(z, 2) <= 3]
        )

        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        assert np.abs(direct_val - new_prob.value) < 1e-4

    def test_with_linear_constraints(self):
        """SOC with additional linear constraints."""
        x = cp.Variable(5)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.norm(x, 2) <= 2, x >= -1, cp.sum(x) >= -3]
        )

        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        assert np.abs(direct_val - new_prob.value) < 1e-4
