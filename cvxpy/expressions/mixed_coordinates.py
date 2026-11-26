"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License. You may obtain
a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize
from cvxpy.problems.problem import Problem


class MixedCoordinates:
    r"""Linear measurements of a positive vector and of its logarithm.

    For a positive vector :math:`x \in \mathbf{R}^n_{++}`, this object exposes
    :math:`Ax` as :attr:`lin` and :math:`B \log x` as :attr:`log`, both as
    affine expressions. If :math:`AB^T = 0`, then as :math:`x` ranges over
    :math:`\mathbf{R}^n_{++}` the pair :math:`(Ax, B\log x)` ranges over exactly
    :math:`A\mathbf{R}^n_{++} \times \operatorname{range}(B)`, so the two
    measurements can be constrained independently and any DCP model in them
    stays DCP. This is the mixed parameterization of an exponential family.

    The vector :math:`x` itself is never an expression, so the condition
    :math:`AB^T = 0` is checked once, here, and nothing else in the problem needs
    to know about it. After solving, :attr:`value` recovers :math:`x`.

    Examples
    --------
    Fit a total concentration and a log ratio of two species:

    >>> c = cp.MixedCoordinates(lin=[[1, 1]], log=[[-1, 1]])
    >>> total, log_ratio = c.lin[0], c.log[0]
    >>> prob = cp.Problem(cp.Minimize(cp.square(total - 3) + cp.square(log_ratio - 0.7)),
    ...                   [total >= 1])
    >>> prob.solve()  # a quadratic program
    >>> c.value       # array([0.995..., 2.004...])

    Parameters
    ----------
    lin : array_like or sparse matrix, optional
        The matrix :math:`A`, with :math:`n` columns.
    log : array_like or sparse matrix, optional
        The matrix :math:`B`, with :math:`n` columns.
    """

    def __init__(self, lin=None, log=None) -> None:
        if lin is None and log is None:
            raise ValueError("MixedCoordinates needs `lin`, `log`, or both.")
        A, B = _as_matrix(lin, "lin"), _as_matrix(log, "log")
        n = (A if A is not None else B).shape[1]
        for name, M in (("lin", A), ("log", B)):
            if M is not None and M.shape[1] != n:
                raise ValueError(f"`{name}` has {M.shape[1]} columns; expected {n}.")
        if A is not None and B is not None:
            _check_orthogonal(A, B)

        self._A, self._B = A, B
        self._p = Variable(n, nonneg=True)  # A @ p covers the closure of A R^n_{++}
        self._z = Variable(n)               # B @ z covers range(B)
        self._lin = A @ self._p if A is not None else None
        self._log = B @ self._z if B is not None else None
        self._cache_key, self._cache_value = None, None

    @property
    def lin(self) -> Expression:
        """The affine expression :math:`Ax`."""
        if self._lin is None:
            raise AttributeError("No `lin` matrix was given.")
        return self._lin

    @property
    def log(self) -> Expression:
        """The affine expression :math:`B\\log x`."""
        if self._log is None:
            raise AttributeError("No `log` matrix was given.")
        return self._log

    @property
    def value(self) -> np.ndarray | None:
        """The positive :math:`x` with the solved values of :attr:`lin` and :attr:`log`.

        When the measurements don't determine :math:`x`, this is the :math:`x`
        closest to :math:`\\exp(z_0)` in KL divergence, where :math:`z_0` is
        the minimum-norm solution of :math:`Bz_0 = B\\log x`. Returns None
        before a solve, and None with a warning if the solved :attr:`lin` lies
        on the boundary of :math:`A\\mathbf{R}^n_{++}`, where no positive
        :math:`x` attains it.
        """
        zeros = np.zeros(self._p.size)
        p = self._p.value if self._A is not None else zeros
        z = self._z.value if self._B is not None else zeros
        if p is None or z is None:
            return None
        key = (p.tobytes(), z.tobytes())
        if key != self._cache_key:
            self._cache_key, self._cache_value = key, self._recover(p, z)
        return self._cache_value

    def _recover(self, p, z) -> np.ndarray | None:
        A, B = self._A, self._B
        n = self._p.size
        # The minimum-norm z0 with B z0 = B z, so x doesn't depend on the solver's z.
        z0 = np.zeros(n) if B is None else np.linalg.lstsq(B, B @ z, rcond=None)[0]
        if A is None:
            return np.exp(z0)
        u = A @ p
        y = _positive_preimage(A, u)
        if y is None:
            warnings.warn(
                "The solved value of `lin` lies on the boundary of the set of "
                "attainable values, so no positive x attains it. "
                "MixedCoordinates.value is None.", stacklevel=3)
            return None
        return _kl_projection(A, u, z0, y)

    def __repr__(self) -> str:
        shapes = ", ".join(f"{name}={M.shape}" for name, M
                           in (("lin", self._A), ("log", self._B)) if M is not None)
        return f"MixedCoordinates({shapes})"


def _as_matrix(M, name):
    if M is None:
        return None
    M = M.toarray() if sp.issparse(M) else np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"`{name}` must be a matrix; got {M.ndim} dimensions.")
    return M


def _check_orthogonal(A, B, tol=1e-9) -> None:
    """Raise unless every row of A is orthogonal to every row of B."""
    scale = np.outer(np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1))
    bad = np.argwhere(np.abs(A @ B.T) > tol * scale)
    if bad.size:
        i, j = bad[0]
        raise ValueError(
            f"Row {i} of `lin` and row {j} of `log` are not orthogonal "
            f"(inner product {A[i] @ B[j]:.3g}). MixedCoordinates requires "
            "lin @ log.T == 0, so that the two measurements are independent.")


def _positive_preimage(A, u, tol=1e-7) -> np.ndarray | None:
    """A point y > 0 with A y = u, or None if there is none."""
    scale = max(np.linalg.norm(u), 1.0)
    y, t = Variable(A.shape[1]), Variable()
    Problem(Maximize(t), [A @ y == u / scale, y >= t, t <= 1]).solve()
    if t.value is None or t.value <= tol:
        return None
    return np.maximum(y.value, t.value) * scale


def _kl_projection(A, u, z0, y, max_iters=100) -> np.ndarray:
    """argmin sum(kl_div(x, exp(z0))) subject to A x = u.

    The solution is x = exp(z0 + A.T @ lam), so log(x) - z0 lies in row(A) and,
    because lin @ log.T == 0, log @ log(x) == log @ z0 holds exactly. Finds lam
    by damped Newton's method on A x(lam) = u, starting from the lam that best
    fits a strictly feasible y.
    """
    lam = np.linalg.lstsq(A.T, np.log(y) - z0, rcond=None)[0]
    tol = 1e-12 * (1 + np.linalg.norm(u))

    def residual(lam):
        return A @ np.exp(z0 + A.T @ lam) - u

    for _ in range(max_iters):
        x = np.exp(z0 + A.T @ lam)
        r = A @ x - u
        if np.linalg.norm(r) <= tol:
            break
        step = np.linalg.lstsq((A * x) @ A.T, -r, rcond=None)[0]
        t = 1.0
        while np.linalg.norm(residual(lam + t * step)) >= np.linalg.norm(r) and t > 1e-8:
            t /= 2
        if t <= 1e-8:
            break  # converged to rounding error
        lam = lam + t * step
    return np.exp(z0 + A.T @ lam)
