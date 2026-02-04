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
import pytest

import cvxpy as cp
from cvxpy.utilities import scopes


class TestQuadFormDPPDetection:
    """Test that is_dpp() correctly identifies DPP-compliant problems."""

    def test_parametric_P_is_dpp_in_scope(self) -> None:
        """quad_form(x, P) with Parameter P is DPP only in quad_form_dpp_scope."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        y = cp.quad_form(x, P)

        # Not DPP without scope (conic path can't handle parametric P)
        assert not y.is_dpp()

        # DPP within scope (QP path can handle it)
        with scopes.quad_form_dpp_scope():
            assert y.is_dpp()

    def test_constant_P_always_dpp(self) -> None:
        """quad_form(x, constant) is always DPP."""
        x = cp.Variable(2)
        P = np.array([[2, 1], [1, 2]])
        assert cp.quad_form(x, P).is_dpp()

    @pytest.mark.parametrize("invalid_case", [
        "param_in_x",      # quad_form(param, P) - quadratic in param
        "x_plus_param",    # quad_form(x + param, P) - x not param-free
        "param_times_param",  # quad_form(x, alpha*P) - quadratic in params
    ])
    def test_invalid_cases_not_dpp(self, invalid_case) -> None:
        """Cases that violate DPP requirements are correctly rejected."""
        x = cp.Variable(2)
        p = cp.Parameter(2)
        P = cp.Parameter((2, 2), PSD=True)
        P.value = np.eye(2)

        if invalid_case == "param_in_x":
            y = cp.quad_form(p, np.eye(2))
        elif invalid_case == "x_plus_param":
            y = cp.quad_form(x + p, P)
        elif invalid_case == "param_times_param":
            alpha = cp.Parameter(nonneg=True)
            alpha.value = 1.0
            y = cp.quad_form(x, alpha * P, assume_PSD=True)

        with scopes.quad_form_dpp_scope():
            assert not y.is_dpp()


class TestQuadFormDPPCompilation:
    """Test that DPP compilation works without P.value set."""

    def test_get_problem_data_without_P_value(self) -> None:
        """BUG: get_problem_data requires P.value to be set.

        This violates the DPP promise that you can compile a problem
        without parameter values. The coeff_extractor.py asserts that
        P.value is not None at line 145.
        """
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        # P.value intentionally NOT set

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])

        # BUG: This should work but raises AssertionError
        with pytest.raises(AssertionError, match="P matrix must be instantiated"):
            prob.get_problem_data(solver=cp.CLARABEL)

    def test_get_problem_data_with_P_value(self) -> None:
        """get_problem_data works when P.value is set and uses DPP path."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        P.value = np.eye(2)

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])
        data, chain, inv_data = prob.get_problem_data(solver=cp.CLARABEL)

        # Verify DPP path is used (no EvalParams)
        reduction_types = [type(r).__name__ for r in chain.reductions]
        assert 'EvalParams' not in reduction_types

    def test_is_dpp_without_P_value(self) -> None:
        """is_dpp() works without P.value set."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        # P.value intentionally NOT set

        with scopes.quad_form_dpp_scope():
            assert cp.quad_form(x, P).is_dpp()


class TestQuadFormDPPCorrectness:
    """Test that DPP path produces correct results."""

    @pytest.mark.parametrize("P_matrix,expected_x", [
        # Diagonal P: min 2*x0^2 + x1^2 s.t. x0+x1=1 => x=[1/3, 2/3]
        (np.array([[2, 0], [0, 1]]), np.array([1/3, 2/3])),
        # Isotropic P: min ||x||^2 s.t. x0+x1=1 => x=[0.5, 0.5]
        (np.eye(2), np.array([0.5, 0.5])),
        # Scaled P: same solution as isotropic
        (5 * np.eye(2), np.array([0.5, 0.5])),
    ])
    def test_minimize_quad_form(self, P_matrix, expected_x) -> None:
        """Minimize quad_form produces correct solution."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        P.value = P_matrix

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])

        # Solve with QP solver
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        x_qp = x.value.copy()

        # Solve with conic solver for comparison
        prob.solve(solver=cp.ECOS)
        assert prob.status == cp.OPTIMAL
        x_conic = x.value.copy()

        # Both should match expected and each other
        assert np.allclose(x_qp, expected_x, rtol=1e-3)
        assert np.allclose(x_conic, expected_x, rtol=1e-3)

    def test_maximize_with_nsd_P(self) -> None:
        """Maximize quad_form with NSD P produces correct solution."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), NSD=True)
        P.value = -np.eye(2)

        # max -||x||^2 s.t. sum(x)=1 => x=[0.5, 0.5], obj=-0.5
        prob = cp.Problem(cp.Maximize(cp.quad_form(x, P)), [cp.sum(x) == 1])

        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], rtol=1e-3)
        assert np.isclose(prob.value, -0.5, rtol=1e-3)

    def test_resolve_with_different_P_values(self) -> None:
        """Re-solving with different P values works correctly."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])

        # First solve: P favors x[1]
        P.value = np.array([[2, 0], [0, 1]])
        prob.solve(solver=cp.CLARABEL)
        x1 = x.value.copy()
        assert x1[1] > x1[0]

        # Second solve: P favors x[0]
        P.value = np.array([[1, 0], [0, 2]])
        prob.solve(solver=cp.CLARABEL)
        x2 = x.value.copy()
        assert x2[0] > x2[1]

        # Verify both are correct
        assert np.allclose(x1, [1/3, 2/3], rtol=1e-3)
        assert np.allclose(x2, [2/3, 1/3], rtol=1e-3)


class TestQuadFormDPPVariants:
    """Test various problem structures with parametric quad_form."""

    def test_quad_form_plus_linear(self) -> None:
        """quad_form + linear term solves correctly."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        c = cp.Parameter(2)
        P.value = np.eye(2)
        c.value = np.array([2, -2])

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + c @ x), [cp.sum(x) == 1])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL

        # Verify with conic solver
        prob.solve(solver=cp.ECOS)
        assert prob.status == cp.OPTIMAL

    def test_quad_form_in_constraint(self) -> None:
        """Parametric quad_form in constraint solves correctly."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        P.value = np.eye(2)

        # min sum(x) s.t. ||x||^2 <= 1 => x = [-1/sqrt(2), -1/sqrt(2)]
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [cp.quad_form(x, P) <= 1])

        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        expected = np.array([-1/np.sqrt(2), -1/np.sqrt(2)])
        assert np.allclose(x.value, expected, rtol=1e-3)

    def test_multiple_quad_forms(self) -> None:
        """Multiple quad_forms with same variable solve correctly."""
        x = cp.Variable(2)
        P1 = cp.Parameter((2, 2), PSD=True)
        P2 = cp.Parameter((2, 2), PSD=True)
        P1.value = np.array([[2, 0], [0, 1]])
        P2.value = np.array([[1, 0], [0, 2]])

        # x'P1x + x'P2x = x'(P1+P2)x = x'[[3,0],[0,3]]x => isotropic
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(x, P1) + cp.quad_form(x, P2)),
            [cp.sum(x) == 1]
        )
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], rtol=1e-3)

    def test_affine_combination_of_params(self) -> None:
        """quad_form(x, P1 + P2) solves correctly."""
        x = cp.Variable(2)
        P1 = cp.Parameter((2, 2), PSD=True)
        P2 = cp.Parameter((2, 2), PSD=True)
        P1.value = np.array([[2, 0], [0, 0]])
        P2.value = np.array([[0, 0], [0, 1]])

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P1 + P2)), [cp.sum(x) == 1])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [1/3, 2/3], rtol=1e-3)


class TestQuadFormDPPEdgeCases:
    """Edge cases for robustness."""

    def test_1x1_matrix(self) -> None:
        """Scalar P (1x1) works correctly."""
        x = cp.Variable(1)
        P = cp.Parameter((1, 1), PSD=True)
        P.value = np.array([[2.0]])

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [x >= 1])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        assert np.isclose(x.value[0], 1.0, rtol=1e-3)
        assert np.isclose(prob.value, 2.0, rtol=1e-3)

    def test_large_matrix(self) -> None:
        """Larger P matrix works correctly."""
        n = 20
        x = cp.Variable(n)
        P = cp.Parameter((n, n), PSD=True)
        P.value = np.eye(n)

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        # Isotropic => uniform distribution
        assert np.allclose(x.value, np.ones(n) / n, rtol=1e-3)

    def test_non_diagonal_P(self) -> None:
        """Non-diagonal P matrix works correctly."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        P.value = np.array([[2, 0.5], [0.5, 1]])

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])

        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        x_qp = x.value.copy()

        prob.solve(solver=cp.ECOS)
        assert np.allclose(x_qp, x.value, rtol=1e-3)


class TestQuadFormDPPKnownLimitations:
    """Document known limitations."""

    def test_alpha_times_P_fails_symmetry_check(self) -> None:
        """KNOWN ISSUE: quad_form(x, alpha * P) fails at construction.

        MulExpression doesn't propagate is_symmetric() property.
        """
        x = cp.Variable(2)
        alpha = cp.Parameter(nonneg=True)
        P = cp.Parameter((2, 2), PSD=True)
        alpha.value = 2.0
        P.value = np.eye(2)

        with pytest.raises(ValueError, match="[Ss]ymmetric|[Hh]ermitian"):
            cp.quad_form(x, alpha * P)

    def test_compilation_requires_P_value(self) -> None:
        """KNOWN ISSUE: DPP compilation requires P.value to be set.

        This violates the DPP promise that compilation should work
        without parameter values. See TestQuadFormDPPCompilation for details.
        """
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        # P.value NOT set

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])

        with pytest.raises(AssertionError):
            prob.get_problem_data(solver=cp.CLARABEL)
