"""
Copyright 2013 Steven Diamond

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

SOLVER = cp.CLARABEL


class TestNormIntegerAxisND:
    """Regression tests for N-D p=2 norms with an integer axis.

    These exercises the per-fiber SOC emission in the p=2 canonicalization:
    N-D SOC constraints with an inner axis could not be lowered to solver
    format (they crashed in ConicSolver.format_constraints).
    """

    def setup_method(self) -> None:
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(3, 4, 5))

    def test_norm2_axis1_3d_solves(self) -> None:
        X = cp.Variable((3, 4, 5))
        prob = cp.Problem(
            cp.Minimize(cp.sum(cp.norm(X, 2, axis=1) - cp.sum(X))),
            [X >= -1, X <= 1],
        )
        prob.solve(solver=SOLVER)
        assert prob.status == cp.OPTIMAL
        ref = np.linalg.norm(X.value, 2, axis=1)
        y = cp.norm(X, 2, axis=1)
        assert y.shape == (3, 5)
        assert np.allclose(y.value, ref, atol=1e-6)

    @pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
    def test_norm2_axis_3d_matches_numpy(self, axis: int) -> None:
        y = cp.norm(cp.Constant(self.X), 2, axis=axis)
        expected = np.linalg.norm(self.X, 2, axis=axis)
        assert y.shape == expected.shape
        assert np.allclose(y.value, expected, atol=1e-10)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_pnorm2_axis_3d_solves(self, axis: int) -> None:
        # Exercises the PnormApprox/SOC canonicalization route directly.
        y = cp.pnorm(cp.Constant(self.X), 2, axis=axis)
        expected = np.linalg.norm(self.X, 2, axis=axis).sum()
        prob = cp.Problem(cp.Minimize(cp.sum(y)))
        prob.solve(solver=SOLVER)
        assert prob.status == cp.OPTIMAL
        assert np.isclose(prob.value, expected, atol=1e-6)

    def test_pnorm2_axis1_gradient(self) -> None:
        X = cp.Variable((3, 4, 5))
        y = cp.norm(X, 2, axis=1)
        X.value = self.X
        G = np.asarray(y.grad[X].todense())
        # CVXPY convention: grad[i, j] = d(y_flat_F[j]) / d(x_flat_F[i]).
        norms = np.linalg.norm(self.X, 2, axis=1)
        expected = np.zeros((60, 15))
        for i in range(3):
            for k in range(5):
                j = i + 3 * k            # y flat (F-order) index of (i, k)
                for r in range(4):
                    m = i + 3 * r + 12 * k  # x flat (F-order) index of (i, r, k)
                    expected[m, j] = self.X[i, r, k] / norms[i, k]
        assert np.allclose(G, expected, atol=1e-10)

    def test_norm2_axis1_3d_keepdims(self) -> None:
        X = cp.Variable((3, 4, 5))
        y = cp.norm(X, 2, axis=1, keepdims=True)
        prob = cp.Problem(cp.Minimize(cp.sum(y) - cp.sum(X)), [X >= 0, X <= 1])
        prob.solve(solver=SOLVER)
        assert prob.status == cp.OPTIMAL
        assert y.shape == (3, 1, 5)
        ref = np.linalg.norm(X.value, 2, axis=1, keepdims=True)
        assert np.allclose(y.value, ref, atol=1e-6)

    def test_perfiber_soc_dual_feasible(self) -> None:
        # The per-fiber SOCs produced by the p=2 canonicalization must
        # recover dual values in the dual second-order cone.
        Xc = cp.Constant(self.X)
        t = cp.Variable((3, 5))
        cons = [cp.SOC(t[i, k], Xc[i, :, k]) for i in range(3) for k in range(5)]
        prob = cp.Problem(cp.Minimize(cp.sum(t)), cons)
        prob.solve(solver=SOLVER)
        assert prob.status == cp.OPTIMAL
        for c in cons:
            rho, lam = np.ravel(c.dual_value[0]), np.ravel(c.dual_value[1])
            assert np.linalg.norm(lam) <= rho[0] + 1e-6


class TestNormOneElementTuple:
    """A one-element axis tuple must be equivalent to an integer axis."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(1)
        self.X = rng.normal(size=(3, 4, 5))

    @pytest.mark.parametrize("p", [1, 2, np.inf])
    def test_one_element_tuple_equivalent_to_int(self, p) -> None:
        Xc = cp.Constant(self.X)
        y_int = cp.norm(Xc, p, axis=1)
        y_tup = cp.norm(Xc, p, axis=(1,))
        assert y_tup.shape == y_int.shape
        assert np.allclose(y_tup.value, y_int.value, atol=1e-10)
        expected = np.linalg.norm(self.X, p, axis=1)
        assert np.allclose(y_tup.value, expected, atol=1e-10)

    @pytest.mark.parametrize("p", [1, np.inf])
    def test_one_element_tuple_solves(self, p) -> None:
        X = cp.Variable((3, 4, 5))
        prob = cp.Problem(
            cp.Minimize(cp.sum(cp.norm(X, p, axis=(1,)))),
            [X >= 0, X <= 1],
        )
        prob.solve(solver=SOLVER)
        assert prob.status == cp.OPTIMAL
        expected = np.linalg.norm(X.value, p, axis=1).sum()
        assert np.isclose(prob.value, expected, atol=1e-6)


class TestMatrixNormTupleAxis:
    """Two-element axis tuples: NumPy matrix-norm semantics per slice."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(2)
        self.X = rng.normal(size=(3, 4, 5))

    def test_p2_is_spectral_not_frobenius(self) -> None:
        y = cp.norm(cp.Constant(self.X), 2, axis=(0, 2))
        expected = np.linalg.norm(self.X, 2, axis=(0, 2))
        fro = np.sqrt(
            (np.abs(self.X) ** 2).sum(axis=(0, 2))
        )
        assert y.shape == (4,)
        # Distinguishes spectral from Frobenius: they differ on generic data.
        assert not np.allclose(expected, fro)
        assert np.allclose(y.value, expected, atol=1e-10)

    def test_fro_tuple(self) -> None:
        y = cp.norm(cp.Constant(self.X), "fro", axis=(0, 2))
        expected = np.sqrt((np.abs(self.X) ** 2).sum(axis=(0, 2)))
        assert y.shape == (4,)
        assert np.allclose(y.value, expected, atol=1e-10)

    def test_nuc_tuple(self) -> None:
        y = cp.norm(cp.Constant(self.X), "nuc", axis=(0, 2))
        expected = np.array(
            [np.linalg.norm(self.X[:, j, :], "nuc") for j in range(4)]
        )
        assert y.shape == (4,)
        assert np.allclose(y.value, expected, atol=1e-10)

    def test_p1_tuple(self) -> None:
        y = cp.norm(cp.Constant(self.X), 1, axis=(0, 2))
        expected = np.array(
            [np.linalg.norm(self.X[:, j, :], 1) for j in range(4)]
        )
        assert y.shape == (4,)
        assert np.allclose(y.value, expected, atol=1e-10)

    def test_pinf_tuple(self) -> None:
        y = cp.norm(cp.Constant(self.X), np.inf, axis=(0, 2))
        expected = np.array(
            [np.linalg.norm(self.X[:, j, :], np.inf) for j in range(4)]
        )
        assert y.shape == (4,)
        assert np.allclose(y.value, expected, atol=1e-10)

    def test_axis_ordering_equivalent(self) -> None:
        a = cp.norm(cp.Constant(self.X), 2, axis=(0, 2))
        b = cp.norm(cp.Constant(self.X), 2, axis=(2, 0))
        assert np.allclose(a.value, b.value, atol=1e-10)

    def test_negative_axes(self) -> None:
        a = cp.norm(cp.Constant(self.X), 2, axis=(0, 2))
        b = cp.norm(cp.Constant(self.X), 2, axis=(-3, -1))
        assert np.allclose(a.value, b.value, atol=1e-10)

    def test_keepdims(self) -> None:
        y = cp.norm(cp.Constant(self.X), 2, axis=(0, 2), keepdims=True)
        assert y.shape == (1, 4, 1)
        expected = np.linalg.norm(self.X, 2, axis=(0, 2))
        assert np.allclose(np.ravel(y.value, order="C"), expected, atol=1e-10)

    def test_2d_full_tuple_is_matrix_norm(self) -> None:
        Xv = np.arange(12, dtype=float).reshape(3, 4)
        Xc = cp.Constant(Xv)
        assert np.isclose(
            cp.norm(Xc, "nuc", axis=(0, 1)).value,
            np.linalg.norm(Xv, "nuc"),
            atol=1e-10,
        )
        y = cp.norm(Xc, 2, axis=(0, 1), keepdims=True)
        assert y.shape == (1, 1)
        assert np.isclose(y.value[0, 0], np.linalg.norm(Xv, 2), atol=1e-10)

    def test_4d_tuple_axis(self) -> None:
        # Two-axis tuples on 4-D input: the result is a 2-D array laid out
        # over the remaining axes (C order), matching NumPy.
        Xv = np.random.default_rng(3).normal(size=(2, 3, 4, 5))
        y = cp.norm(cp.Constant(Xv), 2, axis=(0, 2))
        assert y.shape == (3, 5)
        assert np.allclose(y.value, np.linalg.norm(Xv, 2, axis=(0, 2)), atol=1e-10)

    def test_solve_p2_tuple_matches_numpy(self) -> None:
        X = cp.Variable((3, 4, 5))
        y = cp.norm(X, 2, axis=(0, 2))
        prob = cp.Problem(
            cp.Minimize(cp.sum(y) - cp.sum(X)), [X >= 0, X <= 1]
        )
        prob.solve(solver=SOLVER)
        assert prob.status == cp.OPTIMAL
        ref = np.linalg.norm(X.value, 2, axis=(0, 2))
        assert np.allclose(y.value, ref, atol=1e-6)

    def test_solve_nuc_tuple_matches_numpy(self) -> None:
        X = cp.Variable((3, 4, 5))
        y = cp.norm(X, "nuc", axis=(0, 2))
        prob = cp.Problem(
            cp.Minimize(cp.sum(y) - cp.sum(X)), [X >= 0, X <= 1]
        )
        prob.solve(solver=SOLVER)
        assert prob.status == cp.OPTIMAL
        ref = np.array(
            [np.linalg.norm(X.value[:, j, :], "nuc") for j in range(4)]
        )
        assert np.allclose(y.value, ref, atol=1e-6)


class TestTupleAxisValidation:
    """Invalid axis tuples must raise informative errors."""

    def test_three_axes_rejected(self) -> None:
        X = cp.Constant(np.zeros((2, 3, 4)))
        with pytest.raises(NotImplementedError, match="two axis entries"):
            cp.norm(X, 2, axis=(0, 1, 2))

    def test_duplicate_axes_rejected(self) -> None:
        X = cp.Constant(np.zeros((2, 3, 4)))
        with pytest.raises(ValueError, match="duplicate entries"):
            cp.norm(X, 2, axis=(1, 1))

    def test_empty_tuple_rejected(self) -> None:
        X = cp.Constant(np.zeros((2, 3)))
        with pytest.raises(ValueError, match="empty"):
            cp.norm(X, 2, axis=())

    def test_pnorm_multi_axis_tuple_rejected(self) -> None:
        X = cp.Constant(np.zeros((2, 3, 4)))
        with pytest.raises(ValueError, match="pnorm"):
            cp.pnorm(X, 2, axis=(0, 2))

    def test_norm1_multi_axis_tuple_rejected(self) -> None:
        X = cp.Constant(np.zeros((2, 3, 4)))
        with pytest.raises(ValueError, match="norm1"):
            cp.norm1(X, axis=(0, 2))

    def test_norm_inf_multi_axis_tuple_rejected(self) -> None:
        X = cp.Constant(np.zeros((2, 3, 4)))
        with pytest.raises(ValueError, match="norm_inf"):
            cp.norm_inf(X, axis=(0, 2))

    def test_nd_soc_lowering_still_rejected(self) -> None:
        # The batched N-D SOC lowering is intentionally out of scope (it is
        # a separate issue); pin that a direct 3-D SOC with an inner axis
        # still fails loudly at problem-data time instead of silently
        # producing wrong values. Checked by exception type only: the exact
        # message belongs to the lowering internals.
        t = cp.Variable(12)
        prob = cp.Problem(
            cp.Minimize(cp.sum(t)),
            [cp.SOC(t, cp.Constant(np.zeros((3, 4, 5))), axis=2)],
        )
        with pytest.raises(AssertionError):
            prob.get_problem_data(cp.CLARABEL)


class TestEmptyAxes:
    """Empty-size inputs: NumPy defines every supported matrix norm of an
    empty array as 0, so cvxpy must return a zero constant of the
    NumPy-consistent output shape (also bypassing per-slice construction)."""

    @pytest.mark.parametrize("p", [1, 2, np.inf, "fro", "nuc"])
    def test_empty_remaining_axis(self, p) -> None:
        # No slices exist: the output itself is zero-size.
        Xv = np.zeros((3, 0, 5))
        y = cp.norm(cp.Constant(Xv), p, axis=(0, 2))
        ref = np.linalg.norm(Xv, p, axis=(0, 2))
        assert y.shape == ref.shape == (0,)
        assert np.asarray(y.value).size == 0

    @pytest.mark.parametrize("p", [1, 2, np.inf, "fro", "nuc"])
    def test_empty_remaining_axis_keepdims(self, p) -> None:
        y = cp.norm(cp.Constant(np.zeros((3, 0, 5))), p, axis=(0, 2), keepdims=True)
        ref = np.linalg.norm(np.zeros((3, 0, 5)), p, axis=(0, 2), keepdims=True)
        assert y.shape == ref.shape == (1, 0, 1)

    @pytest.mark.parametrize("p", [1, 2, np.inf, "fro", "nuc"])
    def test_empty_reduced_axes(self, p) -> None:
        # Slices exist but each is (0, 5); every norm is 0. The p='nuc' and
        # p=inf cases also guard against slice-value quirks resurfacing.
        y = cp.norm(cp.Constant(np.zeros((0, 4, 5))), p, axis=(0, 2))
        assert y.shape == (4,)
        assert np.allclose(np.asarray(y.value), np.zeros(4))
