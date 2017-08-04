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

import cvxpy.settings as s
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities.performance_utils as pu
from cvxpy.expressions import cvxtypes
from cvxpy.constraints.nonpos import NonPos


class PSD(NonPos):
    """Constraint X >> 0 that z.TXz >= 0 for all z.
    """

    def __init__(self, expr, constr_id=None):
        # Argument must be square matrix.
        if len(expr.shape) != 2 or expr.shape[0] != expr.shape[1]:
            raise ValueError(
                "Non-square matrix in positive definite constraint."
            )
        super(PSD, self).__init__(expr, constr_id)

    def name(self):
        return "%s >> 0" % self.args[0]

    def is_dcp(self):
        """Both sides must be affine.
        """
        return self.args[0].is_affine()

    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        -------
        Expression
        """
        # TODO wrong. Also requires symmetry.
        min_eig = cvxtypes.lambda_min()(self.args[0] + self.args[0].T)/2
        return cvxtypes.neg()(min_eig)

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Marks the top level constraint as the dual_holder,
        so the dual value will be saved to the Zero.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj, constraints = self.args[0].canonical_form
        dual_holder = PSD(obj, constr_id=self.id)
        return (None, constraints + [dual_holder])

    def format(self, eq_constr, leq_constr, dims, solver):
        """Formats PSD constraints as inequalities for the solver.

        Parameters
        ----------
        eq_constr : list
            A list of the equality constraints in the canonical problem.
        leq_constr : list
            A list of the inequality constraints in the canonical problem.
        dims : dict
            A dict with the dimensions of the conic constraints.
        solver : str
            The solver being called.
        """
        new_leq_constr = self.__format

        # 0 <= A
        leq_constr += new_leq_constr
        # Update dims.
        dims[s.PSD_DIM].append(self.shape[0])

    @pu.lazyprop
    def __format(self):
        """Internal version of format with cached results.

        Returns
        -------
        tuple
            (equality constraints, inequality constraints)
        """
        leq_constr = lu.create_geq(self.expr, constr_id=self.constr_id)
        return [leq_constr]
