"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.constraints.leq_constraint import LeqConstraint
from cvxpy.constraints.semidefinite import SDP
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

class PSDConstraint(LeqConstraint):
    """Constraint X >> Y that z.T(X - Y)z >= 0 for all z.
    """
    OP_NAME = ">>"
    def __init__(self, lh_exp, rh_exp):
        # Arguments must be square matrices or scalars.
        if (lh_exp.size[0] != lh_exp.size[1]) or \
           (rh_exp.size[0] != rh_exp.size[1]):
            raise ValueError(
                "Non-square matrix in positive definite constraint."
            )
        super(PSDConstraint, self).__init__(lh_exp, rh_exp)

    def is_dcp(self):
        """Both sides must be affine.
        """
        return self._expr.is_affine()

    @property
    def value(self):
        """Does the constraint hold?

        Returns
        -------
        bool
        """
        if self._expr.value is None:
            return None
        else:
            mat = self._expr.value
            w, _ = np.linalg.eig(mat + mat.T)
            return w.min()/2 >= -self.TOLERANCE

    @property
    def violation(self):
        """How much is this constraint off by?

        Returns
        -------
        float
        """
        if self._expr.value is None:
            return None
        else:
            mat = self._expr.value
            w, _ = np.linalg.eig(mat + mat.T)
            return -min(w.min()/2, 0)

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Marks the top level constraint as the dual_holder,
        so the dual value will be saved to the EqConstraint.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj, constraints = self._expr.canonical_form
        half = lu.create_const(0.5, (1,1))
        symm = lu.mul_expr(half, lu.sum_expr([obj, lu.transpose(obj)]),
                           obj.size)
        dual_holder = SDP(symm, enforce_sym=False, constr_id=self.id)
        return (None, constraints + [dual_holder])
