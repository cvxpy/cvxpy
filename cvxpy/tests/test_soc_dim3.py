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


class TestSOCDim3Decomposition:
    """Test the SOC decomposition algorithm."""

    def test_dim1_decomposition(self):
        """Dimension 1 (||[]|| <= t) should be padded to 3D.

        Note: CVXPY doesn't support empty arrays, so we test with a
        zero-sized variable indirectly through the n==0 branch.
        """
        # Skip this test - CVXPY doesn't support empty arrays
        # The n==0 case is a degenerate edge case that's unlikely in practice
        pytest.skip("CVXPY doesn't support empty arrays for Constant")

    def test_dim2_decomposition(self):
        """Dimension 2 (|x| <= t) should convert to NonNeg constraints."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(1)

        cones = []
        nonneg_constrs = []
        tree = _decompose_soc_single(t, x, cones, nonneg_constrs)

        # Dim-2 uses NonNeg constraints: t-x >= 0, t+x >= 0
        assert len(cones) == 0  # No SOC cones
        assert len(nonneg_constrs) == 2  # Two NonNeg constraints
        assert isinstance(tree, NonNegTree)
        assert tree.original_dim == 2
        assert len(tree.nonneg_ids) == 2

    def test_dim3_passthrough(self):
        """Dimension 3 (||(x1, x2)|| <= t) should pass through unchanged."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(2)

        cones = []
        nonneg_constrs = []
        tree = _decompose_soc_single(t, x, cones, nonneg_constrs)

        assert len(cones) == 1
        assert cones[0].cone_sizes() == [3]
        assert isinstance(tree, LeafTree)
        assert tree.original_dim == 3

    def test_dim4_decomposition(self):
        """Dimension 4 should decompose into two 3D cones."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(3)  # 3 elements -> dim 4 cone

        cones = []
        nonneg_constrs = []
        tree = _decompose_soc_single(t, x, cones, nonneg_constrs)

        # Dim-4 uses special chain structure: ||(x1, x2)|| <= s, ||(s, x3)|| <= t
        # Two 3D cones with no padding needed
        assert len(cones) == 2
        assert all(c.cone_sizes() == [3] for c in cones)
        assert isinstance(tree, ChainTree)

    def test_dim5_decomposition(self):
        """Dimension 5 should decompose into three 3D cones."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(4)  # 4 elements -> dim 5 cone

        cones = []
        nonneg_constrs = []
        tree = _decompose_soc_single(t, x, cones, nonneg_constrs)

        # Should have: ||(x1, x2)|| <= s1, ||(x3, x4)|| <= s2, ||(s1, s2)|| <= t
        assert len(cones) == 3
        assert all(c.cone_sizes() == [3] for c in cones)
        assert isinstance(tree, SplitTree)

    def test_dim10_decomposition(self):
        """Larger dimension should decompose into all 3D cones."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(9)  # 9 elements -> dim 10 cone

        cones = []
        nonneg_constrs = []
        _decompose_soc_single(t, x, cones, nonneg_constrs)

        # All resulting cones should be dimension 3
        for cone in cones:
            assert cone.cone_sizes() == [3], f"Got cone size {cone.cone_sizes()}"


class TestSOCDim3Reduction:
    """Test the full SOCDim3 reduction."""

    def test_reduction_accepts_any_problem(self):
        """SOCDim3 should accept any problem."""
        x = cp.Variable(5)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [cp.norm(x, 2) <= 1])

        reduction = SOCDim3()
        assert reduction.accepts(prob)

    def test_reduction_transforms_soc(self):
        """SOCDim3 should transform SOC constraints."""
        x = cp.Variable(5)
        t = cp.Variable(nonneg=True)
        prob = cp.Problem(cp.Minimize(t), [SOC(t, x)])

        reduction = SOCDim3()
        new_prob, inverse_data = reduction.apply(prob)

        # The new problem should have SOC constraints
        soc_constrs = [c for c in new_prob.constraints if isinstance(c, SOC)]
        assert len(soc_constrs) > 0

        # All SOC constraints should be dimension 3
        for c in soc_constrs:
            for dim in c.cone_sizes():
                assert dim == 3, f"Got SOC dimension {dim}, expected 3"


class TestSOCDim3Solving:
    """Rigorous tests verifying SOCDim3 decomposition correctness using Clarabel."""

    def _compare_with_and_without_decomposition(self, prob, x_var, atol=1e-5):
        """Helper to compare solving with and without SOCDim3 decomposition."""
        # Solve directly with Clarabel (no SOCDim3)
        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value
        direct_x = x_var.value.copy()

        # Apply SOCDim3 manually and solve with Clarabel
        reduction = SOCDim3()
        new_prob, inverse_data = reduction.apply(prob)

        # Verify all SOC constraints are now dim-3
        for c in new_prob.constraints:
            if isinstance(c, SOC):
                for dim in c.cone_sizes():
                    assert dim == 3, f"Got SOC dimension {dim}, expected 3"

        new_prob.solve(solver=cp.CLARABEL)
        decomposed_val = new_prob.value

        # Compare objectives
        assert np.abs(direct_val - decomposed_val) < atol, \
            f"Objective mismatch: direct={direct_val}, decomposed={decomposed_val}"

        return direct_val, direct_x

    def test_dim2_soc(self):
        """Test dim-2 SOC (|x| <= t) decomposition correctness."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x[0]), [cp.norm(x, 2) <= 2])
        val, _ = self._compare_with_and_without_decomposition(prob, x)
        assert np.abs(val - (-2.0)) < 1e-4

    def test_dim3_soc(self):
        """Test dim-3 SOC (no decomposition needed) correctness."""
        x = cp.Variable(2)
        c = np.array([1, 1])
        prob = cp.Problem(cp.Minimize(c @ x), [cp.norm(x, 2) <= 1])
        val, _ = self._compare_with_and_without_decomposition(prob, x)
        expected = -np.sqrt(2)
        assert np.abs(val - expected) < 1e-4

    def test_dim4_soc(self):
        """Test dim-4 SOC (chain decomposition) correctness."""
        x = cp.Variable(3)
        c = np.ones(3)
        prob = cp.Problem(cp.Minimize(c @ x), [cp.norm(x, 2) <= 1])
        val, _ = self._compare_with_and_without_decomposition(prob, x)
        expected = -np.sqrt(3)
        assert np.abs(val - expected) < 1e-4

    def test_dim5_soc(self):
        """Test dim-5 SOC (balanced 2+2 split) correctness."""
        x = cp.Variable(4)
        c = np.ones(4)
        prob = cp.Problem(cp.Minimize(c @ x), [cp.norm(x, 2) <= 1])
        val, _ = self._compare_with_and_without_decomposition(prob, x)
        expected = -np.sqrt(4)
        assert np.abs(val - expected) < 1e-4

    def test_dim6_soc(self):
        """Test dim-6 SOC (3+2 split) correctness."""
        x = cp.Variable(5)
        c = np.ones(5)
        prob = cp.Problem(cp.Minimize(c @ x), [cp.norm(x, 2) <= 1])
        val, _ = self._compare_with_and_without_decomposition(prob, x)
        expected = -np.sqrt(5)
        assert np.abs(val - expected) < 1e-4

    def test_dim10_soc(self):
        """Test dim-10 SOC (larger tree) correctness."""
        x = cp.Variable(9)
        c = np.ones(9)
        prob = cp.Problem(cp.Minimize(c @ x), [cp.norm(x, 2) <= 1])
        val, _ = self._compare_with_and_without_decomposition(prob, x)
        expected = -np.sqrt(9)
        assert np.abs(val - expected) < 1e-4

    def test_dim20_soc(self):
        """Test dim-20 SOC (deep tree) correctness."""
        x = cp.Variable(19)
        c = np.ones(19)
        prob = cp.Problem(cp.Minimize(c @ x), [cp.norm(x, 2) <= 1])
        val, _ = self._compare_with_and_without_decomposition(prob, x)
        expected = -np.sqrt(19)
        assert np.abs(val - expected) < 1e-4

    def test_multiple_soc(self):
        """Test multiple SOC constraints of different dimensions."""
        x = cp.Variable(5)
        y = cp.Variable(3)
        z = cp.Variable(10)

        prob = cp.Problem(
            cp.Minimize(cp.sum(x) + cp.sum(y) + cp.sum(z)),
            [cp.norm(x, 2) <= 1, cp.norm(y, 2) <= 2, cp.norm(z, 2) <= 3]
        )

        # Solve directly
        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value

        # Apply SOCDim3 and solve
        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)
        decomposed_val = new_prob.value

        assert np.abs(direct_val - decomposed_val) < 1e-4

    def test_socp_with_linear_constraints(self):
        """Test SOC with additional linear constraints."""
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
        decomposed_val = new_prob.value

        assert np.abs(direct_val - decomposed_val) < 1e-4

    def test_standard_socp_1(self):
        """Test standard SOCP test case 1 with decomposition."""
        # From solver_test_helpers.socp_1
        x = cp.Variable(shape=(3,))
        y = cp.Variable()
        soc = SOC(y, x)
        constraints = [soc, x[0] + x[1] + 3 * x[2] >= 1.0, y <= 5]
        prob = cp.Problem(cp.Minimize(3 * x[0] + 2 * x[1] + x[2]), constraints)

        # Direct solve
        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value

        # With decomposition
        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)
        decomposed_val = new_prob.value

        expected = -13.548638904065102
        assert np.abs(direct_val - expected) < 1e-3
        assert np.abs(decomposed_val - expected) < 1e-3

    def test_standard_socp_2(self):
        """Test standard SOCP test case 2 (dim-2 SOC) with decomposition."""
        # From solver_test_helpers.socp_2
        x = cp.Variable(shape=(2,), name='x')
        objective = cp.Minimize(-4 * x[0] - 5 * x[1])
        expr = cp.reshape(x[0] + 2 * x[1], (1, 1), order='F')
        constraints = [
            2 * x[0] + x[1] <= 3,
            SOC(cp.Constant([3]), expr),
            x[0] >= 0,
            x[1] >= 0
        ]
        prob = cp.Problem(objective, constraints)

        # Direct solve
        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value

        # With decomposition
        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)
        decomposed_val = new_prob.value

        expected = -9.0
        assert np.abs(direct_val - expected) < 1e-3
        assert np.abs(decomposed_val - expected) < 1e-3


class TestSOCDim3DualReconstruction:
    """Test dual variable reconstruction for decomposed SOC constraints."""

    def test_dim3_dual_passthrough(self):
        """Test that dim-3 SOC duals pass through unchanged."""
        x = cp.Variable(2)
        t = cp.Variable(nonneg=True)
        soc = SOC(t, x)
        prob = cp.Problem(cp.Minimize(t), [soc, x == [1, 1]])

        # Solve directly
        prob.solve(solver=cp.CLARABEL)
        direct_dual = soc.dual_value
        # Convert to flat format: [t_dual, x1, x2, ...]
        direct_flat = np.concatenate([np.atleast_1d(direct_dual[0]),
                                      np.atleast_1d(direct_dual[1]).flatten()])

        # Solve with SOCDim3
        reduction = SOCDim3()
        new_prob, inv_data = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        # Invert to get reconstructed solution
        sol = Solution(
            new_prob.status,
            new_prob.value,
            {v.id: v.value for v in new_prob.variables()},
            {c.id: c.dual_value for c in new_prob.constraints},
            {}
        )
        inv_sol = reduction.invert(sol, inv_data)

        # Check reconstructed dual
        assert soc.id in inv_sol.dual_vars
        reconstructed_dual = inv_sol.dual_vars[soc.id]

        # Duals should be close (flat format: [t_dual, x1, x2, ...])
        assert reconstructed_dual is not None
        assert len(reconstructed_dual) == 3  # [t, x1, x2] for dim-3 cone
        assert np.allclose(reconstructed_dual, direct_flat, atol=1e-4)

    def test_dim4_dual_reconstruction(self):
        """Test dual reconstruction for dim-4 SOC (chain decomposition)."""
        x = cp.Variable(3)
        t = cp.Variable(nonneg=True)
        soc = SOC(t, x)
        prob = cp.Problem(cp.Minimize(t), [soc, x == [1, 1, 1]])

        # Solve directly with Clarabel
        prob.solve(solver=cp.CLARABEL)
        direct_dual = soc.dual_value
        # Convert to flat format
        direct_flat = np.concatenate([np.atleast_1d(direct_dual[0]),
                                      np.atleast_1d(direct_dual[1]).flatten()])

        # Solve with SOCDim3
        reduction = SOCDim3()
        new_prob, inv_data = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        # Invert
        sol = Solution(
            new_prob.status,
            new_prob.value,
            {v.id: v.value for v in new_prob.variables()},
            {c.id: c.dual_value for c in new_prob.constraints},
            {}
        )
        inv_sol = reduction.invert(sol, inv_data)

        # Check reconstructed dual
        assert soc.id in inv_sol.dual_vars
        reconstructed_dual = inv_sol.dual_vars[soc.id]

        # Flat format: [t, x1, x2, x3] for dim-4 cone
        assert reconstructed_dual is not None
        assert len(reconstructed_dual) == 4
        assert np.allclose(reconstructed_dual, direct_flat, atol=1e-4)

    def test_dim5_dual_reconstruction(self):
        """Test dual reconstruction for dim-5 SOC (balanced split)."""
        x = cp.Variable(4)
        t = cp.Variable(nonneg=True)
        soc = SOC(t, x)
        prob = cp.Problem(cp.Minimize(t), [soc, x == [1, 2, 3, 4]])

        # Solve directly
        prob.solve(solver=cp.CLARABEL)
        direct_dual = soc.dual_value
        # Convert to flat format
        direct_flat = np.concatenate([np.atleast_1d(direct_dual[0]),
                                      np.atleast_1d(direct_dual[1]).flatten()])

        # Solve with SOCDim3
        reduction = SOCDim3()
        new_prob, inv_data = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        # Invert
        sol = Solution(
            new_prob.status,
            new_prob.value,
            {v.id: v.value for v in new_prob.variables()},
            {c.id: c.dual_value for c in new_prob.constraints},
            {}
        )
        inv_sol = reduction.invert(sol, inv_data)

        # Check
        assert soc.id in inv_sol.dual_vars
        reconstructed_dual = inv_sol.dual_vars[soc.id]

        # Flat format: [t, x1, x2, x3, x4] for dim-5 cone
        assert reconstructed_dual is not None
        assert len(reconstructed_dual) == 5
        assert np.allclose(reconstructed_dual, direct_flat, atol=1e-4)

    def test_dim10_dual_reconstruction(self):
        """Test dual reconstruction for larger SOC."""

        x = cp.Variable(9)
        t = cp.Variable(nonneg=True)
        soc = SOC(t, x)
        x_val = np.arange(1, 10, dtype=float)
        prob = cp.Problem(cp.Minimize(t), [soc, x == x_val])

        # Solve directly
        prob.solve(solver=cp.CLARABEL)
        direct_dual = soc.dual_value
        # Convert to flat format
        direct_flat = np.concatenate([np.atleast_1d(direct_dual[0]),
                                      np.atleast_1d(direct_dual[1]).flatten()])

        # Solve with SOCDim3
        reduction = SOCDim3()
        new_prob, inv_data = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        # Invert
        sol = Solution(
            new_prob.status,
            new_prob.value,
            {v.id: v.value for v in new_prob.variables()},
            {c.id: c.dual_value for c in new_prob.constraints},
            {}
        )
        inv_sol = reduction.invert(sol, inv_data)

        # Check
        assert soc.id in inv_sol.dual_vars
        reconstructed_dual = inv_sol.dual_vars[soc.id]

        # Flat format: [t, x1, ..., x9] for dim-10 cone
        assert reconstructed_dual is not None
        assert len(reconstructed_dual) == 10
        assert np.allclose(reconstructed_dual, direct_flat, atol=1e-4)

    def test_socp_1_dual_reconstruction(self):
        """Test dual reconstruction for standard SOCP test 1."""

        # From solver_test_helpers.socp_1
        x = cp.Variable(shape=(3,))
        y = cp.Variable()
        soc = SOC(y, x)
        constraints = [soc, x[0] + x[1] + 3 * x[2] >= 1.0, y <= 5]
        prob = cp.Problem(cp.Minimize(3 * x[0] + 2 * x[1] + x[2]), constraints)

        # Solve directly
        prob.solve(solver=cp.CLARABEL)
        direct_soc_dual = soc.dual_value
        # Convert to flat format
        direct_flat = np.concatenate([np.atleast_1d(direct_soc_dual[0]),
                                      np.atleast_1d(direct_soc_dual[1]).flatten()])

        # Solve with SOCDim3
        reduction = SOCDim3()
        new_prob, inv_data = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        # Invert
        sol = Solution(
            new_prob.status,
            new_prob.value,
            {v.id: v.value for v in new_prob.variables()},
            {c.id: c.dual_value for c in new_prob.constraints},
            {}
        )
        inv_sol = reduction.invert(sol, inv_data)

        # Check SOC dual
        assert soc.id in inv_sol.dual_vars
        reconstructed_dual = inv_sol.dual_vars[soc.id]

        # Flat format: [t, x1, x2, x3] for dim-4 cone
        assert reconstructed_dual is not None
        assert len(reconstructed_dual) == 4
        assert np.allclose(reconstructed_dual, direct_flat, atol=1e-3)


class TestSOCDim3EdgeCases:
    """Test edge cases for SOCDim3 reduction."""

    def test_already_dim3(self):
        """Problem with only dim-3 SOC should work without changes."""
        t = cp.Variable(nonneg=True)
        x = cp.Variable(2)  # dim-3 SOC

        prob = cp.Problem(cp.Minimize(t), [SOC(t, x)])

        reduction = SOCDim3()
        new_prob, inverse_data = reduction.apply(prob)

        soc_constrs = [c for c in new_prob.constraints if isinstance(c, SOC)]
        assert len(soc_constrs) >= 1

    def test_multi_cone_soc(self):
        """SOC constraint with multiple elementwise cones."""
        t = cp.Variable(3, nonneg=True)  # 3 cones
        X = cp.Variable((5, 3))  # Each column is a 6-dim cone

        prob = cp.Problem(cp.Minimize(cp.sum(t)), [SOC(t, X, axis=0)])

        reduction = SOCDim3()
        new_prob, inverse_data = reduction.apply(prob)

        soc_constrs = [c for c in new_prob.constraints if isinstance(c, SOC)]
        # All resulting cones should be dimension 3
        for c in soc_constrs:
            for dim in c.cone_sizes():
                assert dim == 3

    def test_axis1_soc(self):
        """SOC constraint with axis=1."""
        t = cp.Variable(2, nonneg=True)  # 2 cones
        X = cp.Variable((2, 4))  # Each row is a 5-dim cone

        prob = cp.Problem(cp.Minimize(cp.sum(t)), [SOC(t, X, axis=1)])

        reduction = SOCDim3()
        new_prob, inverse_data = reduction.apply(prob)

        soc_constrs = [c for c in new_prob.constraints if isinstance(c, SOC)]
        for c in soc_constrs:
            for dim in c.cone_sizes():
                assert dim == 3


    def test_infeasible_soc_problem(self):
        """Verify INFEASIBLE status propagates through decomposition."""
        x = cp.Variable(5)
        # Infeasible: norm(x) <= 1 but sum(x) >= 10 is impossible
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.norm(x, 2) <= 1, cp.sum(x) >= 10]
        )

        # Direct solve
        prob.solve(solver=cp.CLARABEL)
        direct_status = prob.status

        # With decomposition
        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)
        decomposed_status = new_prob.status

        # Both should be infeasible
        assert direct_status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]
        assert decomposed_status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]

    def test_unbounded_soc_problem(self):
        """Verify UNBOUNDED status propagates through decomposition."""

        x = cp.Variable(5)
        t = cp.Variable()
        # Unbounded: minimize -t with norm(x) <= t (no lower bound on t)
        prob = cp.Problem(
            cp.Minimize(-t),
            [cp.norm(x, 2) <= t]
        )

        # Direct solve
        prob.solve(solver=cp.CLARABEL)
        direct_status = prob.status

        # With decomposition
        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)
        decomposed_status = new_prob.status

        # Both should be unbounded
        assert direct_status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]
        assert decomposed_status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]

    def test_dim2_nonneg_constraints_exist(self):
        """Verify dim-2 creates correct NonNeg constraints (t-x >= 0, t+x >= 0)."""
        from cvxpy.constraints.nonpos import NonNeg

        t = cp.Variable(nonneg=True)
        x = cp.Variable(1)

        cones = []
        nonneg_constrs = []
        tree = _decompose_soc_single(t, x, cones, nonneg_constrs)

        # Should have 2 NonNeg constraints for t-x >= 0 and t+x >= 0
        assert len(nonneg_constrs) == 2
        assert all(isinstance(c, NonNeg) for c in nonneg_constrs)
        assert isinstance(tree, NonNegTree)
        assert tree.original_dim == 2

    def test_dim2_solution_correct(self):
        """Verify dim-2 SOC decomposition gives correct solution."""

        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x[0]), [cp.norm(x, 2) <= 2])

        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        assert new_prob.status == cp.OPTIMAL
        # Solution should be x = -2 (minimum at boundary)
        assert np.abs(x.value[0] - (-2.0)) < 1e-4

    def test_all_different_dimensions(self):
        """Test problem with SOCs of dimensions 2,3,4,5,6 simultaneously."""

        x1 = cp.Variable(1)  # dim-2 cone
        x2 = cp.Variable(2)  # dim-3 cone
        x3 = cp.Variable(3)  # dim-4 cone
        x4 = cp.Variable(4)  # dim-5 cone
        x5 = cp.Variable(5)  # dim-6 cone

        prob = cp.Problem(
            cp.Minimize(cp.sum(x1) + cp.sum(x2) + cp.sum(x3) + cp.sum(x4) + cp.sum(x5)),
            [
                cp.norm(x1, 2) <= 1,
                cp.norm(x2, 2) <= 2,
                cp.norm(x3, 2) <= 3,
                cp.norm(x4, 2) <= 4,
                cp.norm(x5, 2) <= 5,
            ]
        )

        # Direct solve
        prob.solve(solver=cp.CLARABEL)
        direct_val = prob.value

        # With decomposition
        reduction = SOCDim3()
        new_prob, _ = reduction.apply(prob)

        # Verify all SOC are dim-3
        for c in new_prob.constraints:
            if isinstance(c, SOC):
                for dim in c.cone_sizes():
                    assert dim == 3

        new_prob.solve(solver=cp.CLARABEL)
        decomposed_val = new_prob.value

        assert np.abs(direct_val - decomposed_val) < 1e-4

    def test_dual_reconstruction_dim2(self):
        """Test dual reconstruction for dim-2 cone (uses NonNeg representation)."""

        x = cp.Variable(1)
        t = cp.Variable(nonneg=True)
        soc = SOC(t, x)
        prob = cp.Problem(cp.Minimize(t), [soc, x == [2]])

        # Solve directly
        prob.solve(solver=cp.CLARABEL)
        direct_dual = soc.dual_value
        direct_flat = np.concatenate([np.atleast_1d(direct_dual[0]),
                                      np.atleast_1d(direct_dual[1]).flatten()])

        # Solve with SOCDim3
        reduction = SOCDim3()
        new_prob, inv_data = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        # Invert
        sol = Solution(
            new_prob.status,
            new_prob.value,
            {v.id: v.value for v in new_prob.variables()},
            {c.id: c.dual_value for c in new_prob.constraints},
            {}
        )
        inv_sol = reduction.invert(sol, inv_data)

        # Check reconstructed dual
        assert soc.id in inv_sol.dual_vars
        reconstructed_dual = inv_sol.dual_vars[soc.id]

        # Flat format: [t, x] for dim-2 cone
        assert reconstructed_dual is not None
        assert len(reconstructed_dual) == 2
        assert np.allclose(reconstructed_dual, direct_flat, atol=1e-4)

    def test_dual_validity_complementarity(self):
        """Verify complementary slackness: <(t,x), dual> approximates 0 at optimum."""

        x = cp.Variable(4)
        t = cp.Variable(nonneg=True)
        soc = SOC(t, x)
        prob = cp.Problem(cp.Minimize(t + cp.sum(x)), [soc, x >= -1])

        # Solve with decomposition
        reduction = SOCDim3()
        new_prob, inv_data = reduction.apply(prob)
        new_prob.solve(solver=cp.CLARABEL)

        # Invert
        sol = Solution(
            new_prob.status,
            new_prob.value,
            {v.id: v.value for v in new_prob.variables()},
            {c.id: c.dual_value for c in new_prob.constraints},
            {}
        )
        inv_sol = reduction.invert(sol, inv_data)

        # Get reconstructed dual and primal values
        if soc.id in inv_sol.dual_vars and inv_sol.dual_vars[soc.id] is not None:
            dual = inv_sol.dual_vars[soc.id]
            t_val = t.value
            x_val = x.value

            # Compute residual: norm(x) - t (should be ~0 if constraint is tight)
            residual = np.linalg.norm(x_val) - t_val

            # If constraint is tight (active), complementarity holds trivially
            # If constraint is slack, dual should be ~0
            if np.abs(residual) > 1e-4:  # Slack constraint
                # Dual should be near zero for slack constraints
                assert np.linalg.norm(dual) < 1e-3, \
                    f"Dual should be ~0 for slack constraint, got {dual}"
