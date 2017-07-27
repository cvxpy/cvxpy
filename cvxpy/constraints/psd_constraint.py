"""
Copyright 2017 Steven Diamond

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

from cvxpy.expressions import cvxtypes
from cvxpy.constraints.leq_constraint import LeqConstraint
from cvxpy.constraints.semidefinite import SDP
import cvxpy.lin_ops.lin_utils as lu


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
    def residual(self):
        """The residual of the constraint.

        Returns
        -------
        Expression
        """
        min_eig = cvxtypes.lambda_min()(self._expr + self._expr.T)/2
        return cvxtypes.neg()(min_eig)

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Marks the top level constraint as the dual_holder,
        so the dual value will be saved to the EqConstraint.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj, constraints = self._expr.canonical_form
        half = lu.create_const(0.5, (1, 1))
        symm = lu.mul_expr(half, lu.sum_expr([obj, lu.transpose(obj)]),
                           obj.size)
        dual_holder = SDP(symm, enforce_sym=False, constr_id=self.id)
        return (None, constraints + [dual_holder])
