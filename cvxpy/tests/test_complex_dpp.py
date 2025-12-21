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

    def test_dpp_recognition_and_chain(self):
        """Complex parameter problems are DPP and don't use EvalParams."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.abs(x - cp.real(p))))

        # Should be recognized as DPP
        assert prob.is_dpp()

        # get_problem_data should work without parameter value (key DPP feature)
        data, _, _ = prob.get_problem_data(cp.CLARABEL)
        assert data is not None

        # After solve, chain should use Complex2Real but not EvalParams
        p.value = np.array(1 + 2j)
        prob.solve()
        reduction_types = [type(r) for r in prob._cache.solving_chain.reductions]
        assert EvalParams not in reduction_types
        assert Complex2Real in reduction_types

    @pytest.mark.parametrize("shape,param_val,expected", [
        ((), 3 + 4j, 3.0),  # scalar
        ((3,), np.array([1+1j, 2+2j, 3+3j]), np.array([1, 2, 3])),  # vector
        ((2, 2), np.array([[1+1j, 2+2j], [3+3j, 4+4j]]), np.array([[1, 2], [3, 4]])),  # matrix
        ((2, 2, 2), np.ones((2, 2, 2)) * (1+1j), np.ones((2, 2, 2))),  # 3D tensor
    ])
    def test_shapes(self, shape, param_val, expected):
        """DPP works with scalar, vector, matrix, and higher-dim parameters."""
        p = cp.Parameter(shape, complex=True) if shape else cp.Parameter(complex=True)
        x = cp.Variable(shape) if shape else cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= cp.real(p)])

        p.value = np.array(param_val)
        prob.solve()
        if shape:
            assert np.allclose(x.value, expected, atol=1e-4)
        else:
            assert np.isclose(x.value, expected, atol=1e-4)

        # Test fast path re-solve
        p.value = np.array(param_val) * 2
        prob.solve()
        if shape:
            assert np.allclose(x.value, np.array(expected) * 2, atol=1e-4)
        else:
            assert np.isclose(x.value, expected * 2, atol=1e-4)

    @pytest.mark.parametrize("param_type,param_val", [
        ("imag", 3j),  # purely imaginary, use imag()
        ("complex", 3 + 4j),  # complex, use real()
    ])
    def test_param_types(self, param_type, param_val):
        """DPP works with different parameter types."""
        if param_type == "imag":
            p = cp.Parameter(imag=True)
            x = cp.Variable()
            prob = cp.Problem(cp.Minimize(x), [x >= cp.imag(p)])
            p.value = np.array(param_val)
            prob.solve()
            assert np.isclose(x.value, 3.0, atol=1e-4)
        else:
            p = cp.Parameter(complex=True)
            x = cp.Variable()
            prob = cp.Problem(cp.Minimize(x), [x >= cp.real(p)])
            p.value = np.array(param_val)
            prob.solve()
            assert np.isclose(x.value, 3.0, atol=1e-4)

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

        # Fast path re-solve
        p_real.value = 1.0
        p_complex.value = np.array(1 + 1j)
        prob.solve()
        assert np.isclose(x.value, 2.0, atol=1e-4)

    def test_complex_param_with_abs(self):
        """DPP works with complex parameter inside abs."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.abs(x - p)))

        p.value = np.array(3 + 4j)
        prob.solve()
        # Optimal x = real(p) = 3, objective |3 - (3+4j)| = 4
        assert np.isclose(x.value, 3.0, atol=1e-4)
        assert np.isclose(prob.value, 4.0, atol=1e-4)

    @pytest.mark.parametrize("flag,should_have_eval_params", [
        ("enforce_dpp", False),
        ("ignore_dpp", True),
    ])
    def test_dpp_flags(self, flag, should_have_eval_params):
        """enforce_dpp=True succeeds; ignore_dpp=True uses EvalParams."""
        p = cp.Parameter(complex=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= cp.real(p)])
        p.value = np.array(1 + 2j)

        if flag == "enforce_dpp":
            prob.solve(enforce_dpp=True)
        else:
            prob.solve(ignore_dpp=True)

        assert np.isclose(x.value, 1.0, atol=1e-4)
        reduction_types = [type(r) for r in prob._cache.solving_chain.reductions]
        assert (EvalParams in reduction_types) == should_have_eval_params

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_hermitian_param_dpp(self, n):
        """DPP works with Hermitian parameters including fast path re-solve."""
        P = cp.Parameter((n, n), hermitian=True)
        x = cp.Variable()
        # Minimize x such that x*I >= P (i.e., x >= max eigenvalue of P)
        prob = cp.Problem(cp.Minimize(x), [x * np.eye(n) >> P])

        assert prob.is_dpp()

        # get_problem_data works without parameter value
        data, _, _ = prob.get_problem_data(cp.CLARABEL)
        assert data is not None

        # Create random Hermitian matrix and solve
        np.random.seed(n)  # reproducible
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        P_val1 = (A + A.conj().T) / 2
        P.value = P_val1
        prob.solve()
        assert np.isclose(x.value, np.max(np.linalg.eigvalsh(P_val1)), atol=1e-4)

        # Fast path re-solve with different parameter
        P_val2 = P_val1 + np.eye(n)
        P.value = P_val2
        prob.solve()
        assert np.isclose(x.value, np.max(np.linalg.eigvalsh(P_val2)), atol=1e-4)

    def test_hermitian_param_efficient_representation(self):
        """Hermitian parameters use efficient (compact) representation."""
        n = 3
        P = cp.Parameter((n, n), hermitian=True)
        X = cp.Variable((n, n), hermitian=True)
        prob = cp.Problem(cp.Minimize(cp.real(cp.trace(X))), [X >> P])

        P.value = np.array([[1, 1j, 0], [-1j, 1, 1j], [0, -1j, 1]])
        prob.solve()

        c2r = prob._cache.solving_chain.get(Complex2Real)
        assert c2r is not None
        assert P in c2r.canon_methods._parameters
        real_param, imag_param = c2r.canon_methods._parameters[P]

        # Real param: symmetric (n, n); Imag param: compact n*(n-1)//2
        assert real_param.shape == (n, n) and real_param.is_symmetric()
        assert imag_param.shape == (n * (n - 1) // 2,)
