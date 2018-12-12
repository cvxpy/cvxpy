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

import cvxpy.settings as s
import cvxpy.utilities.performance_utils as pu
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.utilities import format_axis
import numpy as np


class SOC(Constraint):
    """A second-order cone constraint for each row/column.

    Assumes ``t`` is a vector the same length as ``X``'s columns (rows) for
    ``axis == 0`` (``1``).

    Attributes:
        t: The scalar part of the second-order constraint.
        X: A matrix whose rows/columns are each a cone.
        axis: Slice by column 0 or row 1.
    """

    def __init__(self, t, X, axis=0, constr_id=None):
        # TODO allow imaginary X.
        assert not t.shape or len(t.shape) == 1
        self.axis = axis
        super(SOC, self).__init__([t, X], constr_id)

    def __str__(self):
        return "SOC(%s, %s)" % (self.args[0], self.args[1])

    @property
    def residual(self):
        t = self.args[0].value
        X = self.args[1].value
        if t is None or X is None:
            return None
        if self.axis == 0:
            X = X.T
        norms = np.linalg.norm(X, ord=2, axis=1)
        zero_indices = np.where(X <= -t)[0]
        averaged_indices = np.where(X >= np.abs(t))[0]
        X_proj = np.array(X)
        t_proj = np.array(t)
        X_proj[zero_indices] = 0
        t_proj[zero_indices] = 0
        avg_coeff = 0.5 * (1 + t/norms)
        X_proj[averaged_indices] = avg_coeff * X[averaged_indices]
        t_proj[averaged_indices] = avg_coeff * t[averaged_indices]
        return np.linalg.norm(np.concatenate([X, t], axis=1) -
                              np.concatenate([X_proj, t_proj], axis=1),
                              ord=2, axis=1)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.axis]

    def format(self, eq_constr, leq_constr, dims, solver):
        """Formats SOC constraints as inequalities for the solver.

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
        leq_constr += self.__format[1]
        # Update dims.
        dims[s.SOC_DIM] += self.cone_sizes()

    @pu.lazyprop
    def __format(self):
        """Internal version of format with cached results.

        Returns
        -------
        tuple
            (equality constraints, inequality constraints)
        """
        return ([], format_axis(self.args[0], self.args[1], self.axis))

    def num_cones(self):
        """The number of elementwise cones.
        """
        return np.prod(self.args[0].shape, dtype=int)

    @property
    def size(self):
        """The number of entries in the combined cones.
        """
        # TODO use size of dual variable(s) instead.
        return sum(self.cone_sizes())

    def cone_sizes(self):
        """The dimensions of the second-order cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        cones = []
        cone_size = 1 + self.args[1].shape[self.axis]
        for i in range(self.num_cones()):
            cones.append(cone_size)
        return cones

    def is_dcp(self):
        """An SOC constraint is DCP if each of its arguments is affine.
        """
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self):
        return False

    # TODO hack
    def canonicalize(self):
        t, t_cons = self.args[0].canonical_form
        X, X_cons = self.args[1].canonical_form
        new_soc = SOC(t, X, self.axis)
        return (None, [new_soc] + t_cons + X_cons)
