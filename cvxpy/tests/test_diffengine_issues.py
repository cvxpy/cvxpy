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

Regressions from the cvxcore issue tracker (discussion #3221), replayed on the
DIFFENGINE canon backend (``ignore_dpp=True``). Each test is a scaled-down
version of a reported bug's reproduction; full-size timings live in
``benchmarks/run_upstream_benchmarks.py``. See
``cvxpy/reductions/solvers/nlp_solvers/diff_engine/ISSUES_3221_STATUS.md``.
"""
import numpy as np
from scipy.linalg import dft

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest

SOLVER = cp.CLARABEL


class TestDiffengineIssues(BaseTest):

    def test_issue_1132_sdp_kron_segfault(self) -> None:
        """Issue #1132: cvxcore segfaulted canonicalizing this SDP at n=300.

        Scaled to n=25; the structure (PSD variable inside kron/reshape/diag
        composites under a Frobenius norm) is what crashed. The diffengine
        path must canonicalize and solve it, matching the CPP backend.
        """
        n = 25
        rng = np.random.default_rng(0)
        points = rng.random((5, n))
        xtx = points.T @ points
        xtxd = np.diag(xtx)
        ones = np.ones(n)
        D = np.outer(ones, xtxd) - 2 * xtx + np.outer(xtxd, ones)
        W = np.ones((n, n))

        def build():
            x = -1 / (n + np.sqrt(n))
            y = -1 / np.sqrt(n)
            V = np.ones((n, n - 1))
            V[0, :] *= y
            V[1:, :] *= x
            V[1:, :] += np.eye(n - 1)
            e = np.ones((n, 1))

            G = cp.Variable((n - 1, n - 1), PSD=True)
            vgv_diag = cp.diag(V @ G @ V.T)
            row = cp.kron(e, cp.reshape(vgv_diag, (1, n), order="F"))
            col = cp.kron(e.T, cp.reshape(vgv_diag, (n, 1), order="F"))
            resid = row + col - 2 * V @ G @ V.T - D
            obj = cp.Maximize(cp.trace(G) - cp.norm(cp.multiply(W, resid), p="fro"))
            return cp.Problem(obj, [])

        prob_de = build()
        prob_de.solve(solver=SOLVER, ignore_dpp=True)
        self.assertEqual(prob_de.status, cp.OPTIMAL)

        prob_base = build()
        prob_base.solve(solver=SOLVER)
        self.assertAlmostEqual(prob_de.value, prob_base.value, places=3)

    def test_issue_2205_kron_unconstrained_qp(self) -> None:
        """Issue #2205: kron(diag(ones), diag(var)) with a complex variable
        sandwiched between dense DFT matrices took minutes to compile.

        Scaled to dim 12. Both the sum_squares and norm2 forms must solve on
        the diffengine path and recover the planted diagonal error exactly
        (the system is consistent by construction).
        """
        rng = np.random.default_rng(0)
        N_r, N_t, N_s = 4, 1, 3
        dim = N_s * N_r * N_t

        # strictly positive input so every diagonal entry of the variable is
        # excited (a binary x can leave the system underdetermined at this scale)
        x = rng.random(dim) + 0.5
        H = dft(dim) * 1j
        H_H = H.conj().T
        err = rng.random(N_r)
        Err = np.kron(np.diag(np.ones(N_s * N_t)), np.diag(err))
        y = H_H @ Err @ H @ x

        for form in (cp.sum_squares, lambda expr: cp.norm2(expr)):
            var = cp.Variable(shape=(N_r,), complex=True)
            Err_est = cp.kron(np.diag(np.ones(N_s * N_t)), cp.diag(var))
            res = form(H_H @ Err_est @ H @ x - y)
            prob = cp.Problem(cp.Minimize(res))
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertItemsAlmostEqual(var.value, err, places=4)

    def test_issue_1043_param_scaled_frobenius(self) -> None:
        """Issue #1043: cvxcore segfaulted on a dictionary-learning problem
        (parameter-scaled squared Frobenius norms) with real data.

        The original .npy data is unavailable; random data with the same
        structure at reduced scale (the original was 312x4096). Two parameter
        combinations exercise the re-solve path; values must match a
        constant-baked baseline.
        """
        rng = np.random.default_rng(0)
        n_samples, d_feat, d_attr = 30, 48, 12
        Xs = rng.standard_normal((n_samples, d_feat))
        Ys = rng.standard_normal((n_samples, d_attr))

        lamda1 = cp.Parameter(nonneg=True)
        lamda2 = cp.Parameter(nonneg=True)
        Ds = cp.Variable(shape=(d_attr, d_feat))
        objective = (
            cp.square(cp.norm(Xs - Ys @ Ds, "fro"))
            + lamda1 * cp.square(cp.norm(Ds, "fro"))
            + lamda2 * (cp.square(cp.norm(Ds, "fro")) - 1)
        )
        prob = cp.Problem(cp.Minimize(objective))

        for lam1, lam2 in [(0.01, 10.0), (10.0, 0.01)]:
            lamda1.value = lam1
            lamda2.value = lam2
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)

            Ds_base = cp.Variable(shape=(d_attr, d_feat))
            base_objective = (
                cp.square(cp.norm(Xs - Ys @ Ds_base, "fro"))
                + lam1 * cp.square(cp.norm(Ds_base, "fro"))
                + lam2 * (cp.square(cp.norm(Ds_base, "fro")) - 1)
            )
            prob_base = cp.Problem(cp.Minimize(base_objective))
            prob_base.solve(solver=SOLVER)
            self.assertAlmostEqual(prob.value, prob_base.value, places=3)

    def test_quantum_kron_partial_transpose_structure(self) -> None:
        """quantum_hilbert_matrix benchmark structure at toy scale: a sparse
        identity kron with a partial transpose, then a constant matmul. Locks
        the sparse-kron active-block path and the block matmul sparsity path
        together.
        """
        from scipy import sparse

        dim = 2
        rng = np.random.default_rng(0)
        X = cp.Variable((dim * dim, dim * dim), symmetric=True)
        eye = sparse.identity(dim, format="csc")
        A = rng.standard_normal((dim, dim))
        AxI = sparse.kron(sparse.csc_matrix(A), sparse.identity(dim * dim, format="csc"))

        expr = cp.kron(eye, cp.partial_transpose(X, dims=[dim, dim], axis=0)) @ AxI
        target = rng.standard_normal(expr.shape)
        obj = cp.Minimize(cp.sum_squares(expr - target))

        prob_de = cp.Problem(obj, [])
        prob_de.solve(solver=SOLVER, ignore_dpp=True)
        self.assertEqual(prob_de.status, cp.OPTIMAL)
        X_de = X.value.copy()

        prob_base = cp.Problem(obj, [])
        prob_base.solve(solver=SOLVER)
        self.assertAlmostEqual(prob_de.value, prob_base.value, places=4)
        self.assertItemsAlmostEqual(X_de, X.value, places=3)
