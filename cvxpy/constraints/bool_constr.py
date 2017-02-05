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
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities.performance_utils as pu
from cvxpy.constraints.constraint import Constraint


class BoolConstr(Constraint):
    """
    A boolean constraint:
        X_{ij} in {0, 1} for all i,j.

    Attributes:
        noncvx_var: A variable constrained to be elementwise boolean.
        lin_op: The linear operator equal to the noncvx_var.
    """
    CONSTR_TYPE = s.BOOL_IDS

    def __init__(self, lin_op):
        self.lin_op = lin_op
        # Create a new nonconvex variable unless the lin_op is a variable.
        if lin_op.type is lo.VARIABLE:
            self.noncvx_var = lin_op
        else:
            self.noncvx_var = lu.create_var(self.lin_op.size)
        super(BoolConstr, self).__init__()

    def __str__(self):
        return "BoolConstr(%s)" % self.lin_op

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
        new_eq = self.__format[0]
        # If an equality constraint was introduced, update eq_constr and dims.
        if new_eq:
            eq_constr += new_eq
            dims[s.EQ_DIM] += self.size[0]*self.size[1]
        # Record the noncvx_var id.
        bool_id = lu.get_expr_vars(self.noncvx_var)[0][0]
        dims[self.CONSTR_TYPE].append(bool_id)

    @pu.lazyprop
    def __format(self):
        """Internal version of format with cached results.

        Returns
        -------
        tuple
            (equality constraints, inequality constraints)
        """
        eq_constr = []
        # If a noncvx var was created, add an equality constraint.
        if self.noncvx_var != self.lin_op:
            eq_constr += lu.create_eq(self.lin_op, self.noncvx_var)
        return (eq_constr, [])

    @property
    def size(self):
        """The dimensions of the semi-definite cone.
        """
        return self.lin_op.size
