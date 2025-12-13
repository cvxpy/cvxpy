"""
Copyright 2024 - the CVXPY Authors

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
from cvxpy.reductions.complex2real.complex2real import Complex2Real
from cvxpy.reductions.eval_params import EvalParams


class TestComplexDPP:
    """Tests for DPP (Disciplined Parameterized Programming) with complex parameters."""

    def test_complex_param_is_dpp(self):
        """Problems with complex parameters should be recognized as DPP."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.abs(x - cp.real(p))))
        assert prob.is_dpp()

    def test_get_problem_data_without_value(self):
        """get_problem_data should work with uninitialized complex parameters."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.abs(x - cp.real(p))))

        # Should not raise - this is the key DPP feature
        data, chain, inv = prob.get_problem_data(cp.CLARABEL)
        assert data is not None

    def test_no_eval_params_in_chain(self):
        """DPP path should not include EvalParams reduction."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.abs(x - cp.real(p))))

        p.value = np.array(1 + 2j)
        prob.solve()

        # EvalParams should NOT be in the chain
        reduction_types = [type(r) for r in prob._cache.solving_chain.reductions]
        assert EvalParams not in reduction_types
        assert Complex2Real in reduction_types

    def test_basic_complex_param_solve(self):
        """Basic solve with complex parameter."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.abs(x - cp.real(p))))

        p.value = np.array(3 + 4j)
        prob.solve()
        assert np.isclose(x.value, 3.0, atol=1e-4)

    def test_fast_path_multiple_solves(self):
        """DPP fast path should work across multiple solves."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= cp.real(p)])

        # First solve
        p.value = np.array(3 + 4j)
        prob.solve()
        assert np.isclose(x.value, 3.0, atol=1e-4)

        # Second solve - should use cached structure
        p.value = np.array(5 + 2j)
        prob.solve()
        assert np.isclose(x.value, 5.0, atol=1e-4)

        # Third solve
        p.value = np.array(-1 + 3j)
        prob.solve()
        assert np.isclose(x.value, -1.0, atol=1e-4)

    def test_purely_imaginary_param(self):
        """DPP works with purely imaginary parameters."""
        p = cp.Parameter(imag=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= cp.imag(p)])

        p.value = np.array(3j)
        prob.solve()
        assert np.isclose(x.value, 3.0, atol=1e-4)

    def test_complex_param_in_objective(self):
        """DPP works with complex parameters in objective."""
        c = cp.Parameter(2, complex=True)
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.real(c.H @ x)), [x >= 0, cp.sum(x) == 1])

        c.value = np.array([1 + 1j, 2 - 1j])
        prob.solve()
        # Optimal x should put weight on the coefficient with smallest real part
        assert prob.status == cp.OPTIMAL

    def test_mixed_real_and_complex_params(self):
        """DPP works with both real and complex parameters."""
        p_real = cp.Parameter()
        p_complex = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= p_real + cp.real(p_complex)])

        p_real.value = 2.0
        p_complex.value = np.array(3 + 4j)
        prob.solve()
        assert np.isclose(x.value, 5.0, atol=1e-4)

        # Fast path
        p_real.value = 1.0
        p_complex.value = np.array(1 + 1j)
        prob.solve()
        assert np.isclose(x.value, 2.0, atol=1e-4)

    def test_enforce_dpp_succeeds(self):
        """enforce_dpp=True should succeed for DPP complex problems."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= cp.real(p)])

        p.value = np.array(1 + 2j)
        prob.solve(enforce_dpp=True)
        assert np.isclose(x.value, 1.0, atol=1e-4)

    def test_vector_complex_param(self):
        """DPP works with vector complex parameters."""
        p = cp.Parameter(3, complex=True)
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= cp.real(p)])

        p.value = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        prob.solve()
        assert np.allclose(x.value, [1, 2, 3], atol=1e-4)

    def test_matrix_complex_param(self):
        """DPP works with matrix complex parameters."""
        P = cp.Parameter((2, 2), complex=True)
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= cp.real(P) @ np.ones(2)])

        P.value = np.array([[1 + 1j, 0], [0, 2 + 2j]])
        prob.solve()
        # x >= [1, 2]
        assert np.allclose(x.value, [1, 2], atol=1e-4)

    def test_complex_param_with_abs(self):
        """DPP works with complex parameter inside abs."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        # minimize |x - p| where p is complex
        prob = cp.Problem(cp.Minimize(cp.abs(x - p)))

        p.value = np.array(3 + 4j)
        prob.solve()
        # Optimal x = real(p) = 3, with objective |3 - (3+4j)| = 4
        assert np.isclose(x.value, 3.0, atol=1e-4)
        assert np.isclose(prob.value, 4.0, atol=1e-4)

    def test_backward_compat_ignore_dpp(self):
        """ignore_dpp=True still works with complex parameters."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= cp.real(p)])

        p.value = np.array(1 + 2j)
        prob.solve(ignore_dpp=True)
        assert np.isclose(x.value, 1.0, atol=1e-4)

        # EvalParams should be in the chain when ignore_dpp=True
        reduction_types = [type(r) for r in prob._cache.solving_chain.reductions]
        assert EvalParams in reduction_types


@pytest.mark.skipif(cp.DIFFCP not in cp.installed_solvers(), reason="diffcp not installed")
class TestComplexDPPDerivatives:
    """Tests for backward/forward differentiation with complex parameters."""

    def test_backward_real_part(self):
        """Backward differentiation through real part of complex parameter."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        # minimize (x - real(p))^2, optimal x* = real(p)
        prob = cp.Problem(cp.Minimize(cp.square(x - cp.real(p))))

        p.value = np.array(3.0 + 4.0j)
        prob.solve(requires_grad=True)
        assert np.isclose(x.value, 3.0, atol=1e-3)

        # Backward: set x.gradient = 1 to get dx*/dp
        x.gradient = 1.0
        prob.backward()

        # dx*/d(real(p)) = 1, dx*/d(imag(p)) = 0
        # So p.gradient should be 1 + 0j
        assert np.isclose(np.real(p.gradient), 1.0, atol=1e-3)
        assert np.isclose(np.imag(p.gradient), 0.0, atol=1e-3)

    def test_backward_imag_part(self):
        """Backward differentiation through imag part of complex parameter."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        # minimize (x - imag(p))^2, optimal x* = imag(p)
        prob = cp.Problem(cp.Minimize(cp.square(x - cp.imag(p))))

        p.value = np.array(3.0 + 4.0j)
        prob.solve(requires_grad=True)
        assert np.isclose(x.value, 4.0, atol=1e-3)

        x.gradient = 1.0
        prob.backward()

        # dx*/d(real(p)) = 0, dx*/d(imag(p)) = 1
        # So p.gradient should be 0 + 1j
        assert np.isclose(np.real(p.gradient), 0.0, atol=1e-3)
        assert np.isclose(np.imag(p.gradient), 1.0, atol=1e-3)

    def test_backward_both_parts(self):
        """Backward differentiation using both real and imag parts."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        y = cp.Variable()
        # minimize (x - real(p))^2 + (y - imag(p))^2
        prob = cp.Problem(
            cp.Minimize(cp.square(x - cp.real(p)) + cp.square(y - cp.imag(p)))
        )

        p.value = np.array(3.0 + 4.0j)
        prob.solve(requires_grad=True)
        assert np.isclose(x.value, 3.0, atol=1e-3)
        assert np.isclose(y.value, 4.0, atol=1e-3)

        x.gradient = 1.0
        y.gradient = 1.0
        prob.backward()

        # dx*/d(real(p)) = 1, dy*/d(imag(p)) = 1
        # p.gradient = 1 + 1j
        assert np.isclose(np.real(p.gradient), 1.0, atol=1e-3)
        assert np.isclose(np.imag(p.gradient), 1.0, atol=1e-3)

    def test_forward_real_part(self):
        """Forward differentiation (derivative) through real part."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= cp.real(p)])

        p.value = np.array(3.0 + 4.0j)
        prob.solve(requires_grad=True)
        assert np.isclose(x.value, 3.0, atol=1e-3)

        # Perturb only the real part
        p.delta = 1.0 + 0.0j
        prob.derivative()

        # x* = real(p), so dx*/d(real(p)) = 1
        assert np.isclose(x.delta, 1.0, atol=1e-3)

    def test_forward_imag_part(self):
        """Forward differentiation (derivative) through imag part."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= cp.imag(p)])

        p.value = np.array(3.0 + 4.0j)
        prob.solve(requires_grad=True)
        assert np.isclose(x.value, 4.0, atol=1e-3)

        # Perturb only the imaginary part
        p.delta = 0.0 + 1.0j
        prob.derivative()

        # x* = imag(p), so dx*/d(imag(p)) = 1
        assert np.isclose(x.delta, 1.0, atol=1e-3)

    def test_forward_complex_delta(self):
        """Forward differentiation with complex delta."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(x + y), [x >= cp.real(p), y >= cp.imag(p)])

        p.value = np.array(3.0 + 4.0j)
        prob.solve(requires_grad=True)
        assert np.isclose(x.value, 3.0, atol=1e-3)
        assert np.isclose(y.value, 4.0, atol=1e-3)

        # Perturb both parts
        p.delta = 2.0 + 3.0j
        prob.derivative()

        # x* = real(p), y* = imag(p)
        assert np.isclose(x.delta, 2.0, atol=1e-3)
        assert np.isclose(y.delta, 3.0, atol=1e-3)
