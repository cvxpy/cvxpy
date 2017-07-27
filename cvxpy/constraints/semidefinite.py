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

import cvxpy.settings as s
from cvxpy.error import SolverError
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities.performance_utils as pu
from cvxpy.constraints.constraint import Constraint
import numpy as np
import scipy.sparse as sp


class SDP(Constraint):
    """
    A semi-definite cone constraint:
        { symmetric A | x.T*A*x >= 0 for all x }
    (the set of all symmetric matrices such that the quadratic
    form x.T*A*x is positive for all x).

    Attributes:
        A: The matrix variable constrained to be semi-definite.
        enforce_sym: Should symmetry constraints be added?
        constr_id: The id assigned to the inequality constraint.
    """

    def __init__(self, A, enforce_sym=True, constr_id=None):
        self.A = A
        self.enforce_sym = enforce_sym
        super(SDP, self).__init__(constr_id)

    def __str__(self):
        return "SDP(%s)" % self.A

    def format(self, eq_constr, leq_constr, dims, solver):
        """Formats SDP constraints as inequalities for the solver.

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
        if solver.name() in [s.CVXOPT, s.MOSEK]:
            new_eq_constr, new_leq_constr = self.__CVXOPT_format
        elif solver.name() == s.SCS:
            new_eq_constr, new_leq_constr = self.__SCS_format
        else:
            raise SolverError(
                "Solver does not support positive semidefinite cone."
            )

        if self.enforce_sym:
            # upper_tri(A) == upper_tri(A.T)
            eq_constr += new_eq_constr
            # Update dims.
            dims[s.EQ_DIM] += (self.size[0]*(self.size[1] - 1))//2
        # 0 <= A
        leq_constr += new_leq_constr
        # Update dims.
        dims[s.SDP_DIM].append(self.size[0])

    @pu.lazyprop
    def __SCS_format(self):
        eq_constr = self._get_eq_constr()
        term = self._scaled_lower_tri()
        if self.constr_id is None:
            leq_constr = lu.create_geq(term)
        else:
            leq_constr = lu.create_geq(term, constr_id=self.constr_id)
        return ([eq_constr], [leq_constr])

    def _scaled_lower_tri(self):
        """Returns a LinOp representing the lower triangular entries.

            Scales the strictly lower triangular entries by
            sqrt(2) as required by SCS.
        """
        rows = cols = self.size[0]
        entries = rows*(cols + 1)//2
        val_arr = []
        row_arr = []
        col_arr = []
        count = 0
        for j in range(cols):
            for i in range(rows):
                if j <= i:
                    # Index in the original matrix.
                    col_arr.append(j*rows + i)
                    # Index in the extracted vector.
                    row_arr.append(count)
                    if j == i:
                        val_arr.append(1.0)
                    else:
                        val_arr.append(np.sqrt(2))
                    count += 1

        size = (entries, rows*cols)
        coeff = sp.coo_matrix((val_arr, (row_arr, col_arr)), size).tocsc()
        coeff = lu.create_const(coeff, size, sparse=True)
        vect = lu.reshape(self.A, (rows*cols, 1))
        return lu.mul_expr(coeff, vect, (entries, 1))

    @pu.lazyprop
    def __CVXOPT_format(self):
        """Internal version of format with cached results.

        Returns
        -------
        tuple
            (equality constraints, inequality constraints)
        """
        eq_constr = self._get_eq_constr()
        if self.constr_id is None:
            leq_constr = lu.create_geq(self.A)
        else:
            leq_constr = lu.create_geq(self.A, constr_id=self.constr_id)
        return ([eq_constr], [leq_constr])

    def _get_eq_constr(self):
        """Returns the equality constraints for the SDP constraint.
        """
        upper_tri = lu.upper_tri(self.A)
        lower_tri = lu.upper_tri(lu.transpose(self.A))
        return lu.create_eq(upper_tri, lower_tri)

    @property
    def size(self):
        """The dimensions of the semidefinite cone.
        """
        return self.A.size
