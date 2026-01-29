"""
Copyright 2025, the CVXPY Authors

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
import scipy.sparse as sp

from cvxpy.constraints import NonNeg, Zero


class AffineQpMixin:
    """Mixin for conic solvers that only support affine constraints (QP solvers).

    This mixin allows QP-only solvers to be used through the conic interface.
    Solvers using this mixin will:
    - Only accept Zero (equality) and NonNeg (inequality) constraints
    - Support quadratic objectives (P matrix)
    - Convert the stacked conic format to split QP format

    The conic format is:
        minimize    (1/2) x' P x + c' x
        subject to  A x + b in K
        where K = Zero x NonNeg (stacked)

    The QP format is:
        minimize    (1/2) x' P x + q' x
        subject to  A_eq x = b_eq     (from Zero cone)
                    F x <= g           (from NonNeg cone)
    """

    SUPPORTED_CONSTRAINTS = [Zero, NonNeg]

    def supports_quad_obj(self) -> bool:
        """QP solvers support quadratic objectives."""
        return True

    @staticmethod
    def conic_to_qp_format(data, cone_dims):
        """Convert stacked conic (A, b) to split QP (A_eq, b_eq, F, g).

        Parameters
        ----------
        data : dict
            Conic problem data with keys 'A', 'b', 'c', and optionally 'P'.
            - A: constraint matrix (negative of conic A, i.e., A x <= b form)
            - b: constraint vector
        cone_dims : ConeDims
            Cone dimensions containing .zero (equality) and .nonneg (inequality).

        Returns
        -------
        dict
            QP problem data with keys:
            - P: quadratic cost matrix (if present)
            - q: linear cost vector
            - A: equality constraint matrix
            - b: equality RHS
            - F: inequality constraint matrix
            - g: inequality RHS
        """
        import cvxpy.settings as s

        # Get dimensions
        # Note: even if A has no rows, we still need the number of variables
        n = data[s.A].shape[1]
        len_eq = cone_dims.zero
        len_ineq = cone_dims.nonneg

        # The conic format from ConicSolver.apply() is:
        #   A_conic x + s = b, s in cone
        # Which means for Zero cone: A_conic x = b (equality: -A_conic x + b = 0)
        # And for NonNeg cone: A_conic x + s = b, s >= 0 => A_conic x <= b
        #
        # But ConicSolver.apply() stores -A (negated), so:
        #   data[A] = -A_conic => -data[A] x = b for equality
        #   data[A] x <= b for inequality
        #
        # QP format is:
        #   A_eq x = b_eq
        #   F x <= g

        A_full = data[s.A]  # This is -A_conic from the formulation
        b_full = data[s.B]

        # Split by cone type (equality first, then inequality)
        if len_eq > 0:
            # For equality: we have -A_conic, and want A_eq x = b_eq
            # From -A_conic x = b => A_eq = -A_conic = data[A], b_eq = b
            A_eq = A_full[:len_eq, :]
            b_eq = b_full[:len_eq]
        else:
            A_eq = sp.csr_array((0, n))
            b_eq = np.array([])

        if len_ineq > 0:
            # For inequality: we have -A_conic x + s = b, s >= 0
            # This means -A_conic x <= b
            # So F = -A_conic = data[A], g = b
            F = A_full[len_eq:, :]
            g = b_full[len_eq:]
        else:
            F = sp.csr_array((0, n))
            g = np.array([])

        qp_data = {
            s.Q: data[s.C],  # Linear cost
            s.A: sp.csc_array(A_eq),
            s.B: b_eq,
            s.F: sp.csc_array(F),
            s.G: g,
        }

        if s.P in data:
            qp_data[s.P] = sp.csc_array(data[s.P])

        return qp_data

    @staticmethod
    def conic_to_osqp_format(data, cone_dims):
        """Convert stacked conic format to OSQP/QPALM format: l <= Ax <= u.

        This is more efficient than conic_to_qp_format() for solvers that use
        the l <= Ax <= u constraint format, as it avoids splitting and
        re-stacking the constraint matrix.

        Parameters
        ----------
        data : dict
            Conic problem data with keys 'A', 'b', 'c', and optionally 'P'.
        cone_dims : ConeDims
            Cone dimensions containing .zero (equality) and .nonneg (inequality).

        Returns
        -------
        dict
            Problem data with keys:
            - P: quadratic cost matrix (if present), as csc_array
            - q: linear cost vector
            - A: full constraint matrix, as csc_array
            - l: lower bounds (b for equality, -inf for inequality)
            - u: upper bounds (b for both)
        """
        import cvxpy.settings as s

        len_eq = cone_dims.zero
        len_ineq = cone_dims.nonneg

        A = sp.csc_array(data[s.A])
        b = data[s.B]
        q = data[s.C]

        # Build lower and upper bound vectors:
        # - Equality rows (0 to len_eq-1): lower = upper = b
        # - Inequality rows (len_eq to end): lower = -inf, upper = b
        upper = b.copy()
        lower = np.concatenate([b[:len_eq], -np.inf * np.ones(len_ineq)])

        result = {
            s.Q: q,
            s.A: A,
            'l': lower,
            'u': upper,
        }

        if s.P in data:
            result[s.P] = sp.csc_array(data[s.P])

        return result
