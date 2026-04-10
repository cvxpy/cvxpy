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

SOLVER = cp.CLARABEL
RTOL = 1e-3


def _solve(problem: cp.Problem, **kwargs) -> None:
    problem.solve(solver=SOLVER, **kwargs)
    assert problem.status == cp.OPTIMAL


def _solve_and_copy(problem: cp.Problem, variable: cp.Variable, **kwargs) -> np.ndarray:
    _solve(problem, **kwargs)
    return variable.value.copy()


def _assert_allclose(actual, expected) -> None:
    assert np.allclose(actual, expected, rtol=RTOL)


def _assert_qp_matches_conic(problem: cp.Problem, variable: cp.Variable) -> np.ndarray:
    qp_solution = _solve_and_copy(problem, variable)
    _solve(problem, use_quad_obj=False)
    _assert_allclose(qp_solution, variable.value)
    return qp_solution


def _assert_resolve_matches_fresh(
    first_solution: np.ndarray,
    resolved_solution: np.ndarray,
    fresh_problem: cp.Problem,
    variable: cp.Variable,
) -> None:
    fresh_solution = _solve_and_copy(fresh_problem, variable)
    assert not np.allclose(first_solution, resolved_solution, atol=RTOL)
    _assert_allclose(resolved_solution, fresh_solution)


class TestQuadFormDPPDetection:
    """Test that is_dpp() correctly identifies DPP-compliant problems."""

    def test_parametric_P_is_dpp_in_scope(self) -> None:
        """quad_form(x, P) with Parameter P is DPP only in quad_form_dpp_scope."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        y = cp.quad_form(x, P)

        assert not y.is_dpp()
        with scopes.quad_form_dpp_scope():
            assert y.is_dpp()

    def test_constant_P_always_dpp(self) -> None:
        """quad_form(x, constant) is always DPP."""
        x = cp.Variable(2)
        P = np.array([[2, 1], [1, 2]])
        assert cp.quad_form(x, P).is_dpp()

    @pytest.mark.parametrize("invalid_case", [
        "param_in_x",
        "x_plus_param",
        "param_times_param",
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
        """get_problem_data works without P.value set (DPP promise)."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])
        _, chain, _ = prob.get_problem_data(solver=cp.CLARABEL)

        reduction_types = [type(r).__name__ for r in chain.reductions]
        assert 'EvalParams' not in reduction_types

    def test_is_dpp_without_P_value(self) -> None:
        """is_dpp() works without P.value set."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)

        with scopes.quad_form_dpp_scope():
            assert cp.quad_form(x, P).is_dpp()


class TestQuadFormDPPCorrectness:
    """Test that DPP path produces correct results."""

    @pytest.mark.parametrize(
        ("P_matrix", "expected_x"),
        [
            pytest.param(np.array([[2, 0], [0, 1]]), np.array([1 / 3, 2 / 3]), id="diagonal"),
            pytest.param(np.eye(2), np.array([0.5, 0.5]), id="identity"),
            pytest.param(5 * np.eye(2), np.array([0.5, 0.5]), id="scaled_identity"),
            pytest.param(
                np.array([[2, 0.5], [0.5, 1]]),
                None,
                id="dense_psd",
            ),
        ],
    )
    def test_minimize_quad_form(self, P_matrix, expected_x) -> None:
        """Minimizing a parametric quad_form matches the conic path."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        P.value = P_matrix

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])
        qp_solution = _assert_qp_matches_conic(prob, x)

        if expected_x is not None:
            _assert_allclose(qp_solution, expected_x)

    def test_maximize_with_nsd_P(self) -> None:
        """Maximize quad_form with NSD P produces correct solution."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), NSD=True)
        P.value = -np.eye(2)

        prob = cp.Problem(cp.Maximize(cp.quad_form(x, P)), [cp.sum(x) == 1])
        _solve(prob)

        _assert_allclose(x.value, [0.5, 0.5])
        assert np.isclose(prob.value, -0.5, rtol=RTOL)

    def test_resolve_with_different_P_values(self) -> None:
        """Re-solving with different P values works correctly."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [cp.sum(x) == 1])

        P.value = np.array([[2, 0], [0, 1]])
        _solve(prob)
        _assert_allclose(x.value, [1 / 3, 2 / 3])

        P.value = np.array([[1, 0], [0, 2]])
        _solve(prob)
        _assert_allclose(x.value, [2 / 3, 1 / 3])


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
        _solve(prob)
        _assert_allclose(x.value, [-0.5, 1.5])

    def test_quad_form_plus_constant_linear(self) -> None:
        """quad_form(x, P) + c @ x with constant c solves correctly.

        Regression test: ensures the q coefficient from the linear term is
        not overwritten when the quad_form dummy variable is processed after
        the true variable in extract_quadratic_coeffs.
        """
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        P.value = np.eye(2)
        c = np.array([2.0, -2.0])

        # min x'Px + c'x s.t. sum(x) == 1
        # Analytic: x* = (1 - P^{-1} c) / (1' P^{-1} 1) adjusted for constraint
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + c @ x), [cp.sum(x) == 1])
        qp_solution = _assert_qp_matches_conic(prob, x)
        _assert_allclose(qp_solution, [-0.5, 1.5])

        # Re-solve with a different P to ensure DPP caching works.
        P.value = np.array([[2, 0], [0, 1]])
        _assert_qp_matches_conic(prob, x)

    def test_quad_form_in_constraint(self) -> None:
        """Parametric quad_form in constraint solves correctly."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        P.value = np.eye(2)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [cp.quad_form(x, P) <= 1])
        _solve(prob)
        expected = np.array([-1 / np.sqrt(2), -1 / np.sqrt(2)])
        _assert_allclose(x.value, expected)

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
        _solve(prob)
        _assert_allclose(x.value, [0.5, 0.5])

    def test_affine_combination_of_params(self) -> None:
        """quad_form(x, P1 + P2) solves correctly."""
        x = cp.Variable(2)
        P1 = cp.Parameter((2, 2), PSD=True)
        P2 = cp.Parameter((2, 2), PSD=True)
        P1.value = np.array([[2, 0], [0, 0]])
        P2.value = np.array([[0, 0], [0, 1]])

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P1 + P2)), [cp.sum(x) == 1])
        _solve(prob)
        _assert_allclose(x.value, [1 / 3, 2 / 3])

    def test_resolve_scaled_param(self) -> None:
        """Re-solving quad_form(x, 2*P) updates P correctly."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)

        prob = cp.Problem(
            cp.Minimize(cp.quad_form(x, 2 * P) + 5 * x[0] + 3 * x[1])
        )

        P.value = np.array([[2, 0], [0, 1]])
        first_solution = _solve_and_copy(prob, x)

        P.value = np.array([[1, 0], [0, 5]])
        resolved_solution = _solve_and_copy(prob, x)

        # Verify against a fresh solve with the second P value.
        fresh_problem = cp.Problem(
            cp.Minimize(cp.quad_form(x, 2 * P) + 5 * x[0] + 3 * x[1])
        )
        _assert_resolve_matches_fresh(first_solution, resolved_solution, fresh_problem, x)

    def test_resolve_affine_combination_of_params(self) -> None:
        """Re-solving quad_form(x, P1 + P2) updates correctly."""
        x = cp.Variable(2)
        P1 = cp.Parameter((2, 2), PSD=True)
        P2 = cp.Parameter((2, 2), PSD=True)

        prob = cp.Problem(
            cp.Minimize(cp.quad_form(x, P1 + P2)), [cp.sum(x) == 1]
        )

        # P1+P2 = [[2,0],[0,1]] — weights x[0] more, so x[1] preferred.
        P1.value = np.array([[2, 0], [0, 0]])
        P2.value = np.array([[0, 0], [0, 1]])
        _solve(prob)
        _assert_allclose(x.value, [1 / 3, 2 / 3])

        # P1+P2 = [[1,0],[0,2]] — weights x[1] more, so x[0] preferred.
        P1.value = np.array([[1, 0], [0, 0]])
        P2.value = np.array([[0, 0], [0, 2]])
        _solve(prob)
        _assert_allclose(x.value, [2 / 3, 1 / 3])


class TestQuadFormDPPConstraintResolve:
    """Regression tests: parametric quad_form in constraints must update on re-solve.

    When quad_form(x, P) appears in a constraint (not the objective), the conic
    canonicalizer bakes in numeric Cholesky factors from the first P.value.
    If the problem is incorrectly classified as DPP, re-solving with a different
    P.value reuses stale factors and produces wrong results.
    """

    def test_constraint_quad_form_resolve(self) -> None:
        """Re-solving with different P in constraint must update the constraint."""
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [cp.quad_form(x, P) <= 1])

        P.value = np.eye(2)
        first_solution = _solve_and_copy(prob, x)
        _assert_allclose(first_solution, [-1 / np.sqrt(2), -1 / np.sqrt(2)])

        # Tighter on x[0]: solution should shift toward x[1].
        P.value = np.array([[4, 0], [0, 1]])
        resolved_solution = _solve_and_copy(prob, x)

        # Cross-check against a fresh problem with the same P.
        fresh_problem = cp.Problem(cp.Minimize(cp.sum(x)), [cp.quad_form(x, P) <= 1])
        _assert_resolve_matches_fresh(first_solution, resolved_solution, fresh_problem, x)

    def test_mixed_objective_and_constraint_resolve(self) -> None:
        """Parametric P in both objective and constraint must both update."""
        x = cp.Variable(2)
        P_obj = cp.Parameter((2, 2), PSD=True)
        P_constr = cp.Parameter((2, 2), PSD=True)

        prob = cp.Problem(
            cp.Minimize(cp.quad_form(x, P_obj)),
            [cp.quad_form(x, P_constr) <= 4, cp.sum(x) == 1]
        )

        P_obj.value = np.eye(2)
        P_constr.value = np.eye(2)
        first_solution = _solve_and_copy(prob, x)

        # Change constraint P only — tighter on x[0].
        P_constr.value = np.array([[100, 0], [0, 1]])
        resolved_solution = _solve_and_copy(prob, x)

        # Cross-check against a fresh problem.
        fresh_problem = cp.Problem(
            cp.Minimize(cp.quad_form(x, P_obj)),
            [cp.quad_form(x, P_constr) <= 4, cp.sum(x) == 1]
        )
        _assert_resolve_matches_fresh(first_solution, resolved_solution, fresh_problem, x)


def test_hermitian_param_resolve() -> None:
    """Test quad_form DPP re-solve with complex hermitian P."""
    x = cp.Variable(2, complex=True)
    P = cp.Parameter((2, 2), PSD=True, complex=True)
    constrs = [cp.real(x[0]) >= 1, cp.real(x[1]) >= 1]
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), constrs)

    P.value = np.array([[2, 1 + 1j], [1 - 1j, 3]])
    first_solution = _solve_and_copy(prob, x)

    P.value = np.array([[4, 1 + 2j], [1 - 2j, 5]])
    resolved_solution = _solve_and_copy(prob, x)

    assert not np.allclose(first_solution, resolved_solution, atol=RTOL)
    _solve(prob, use_quad_obj=False)
    _assert_allclose(resolved_solution, x.value)
