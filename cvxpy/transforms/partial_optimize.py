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

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.lin_ops.lin_op as lo
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.problems.problem import Problem
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.expression import Expression
import copy

def partial_optimize(prob, opt_vars=None, dont_opt_vars=None):
    """Partially optimizes the given problem over the specified variables.

    Either opt_vars or dont_opt_vars must be given.
    If both are given, they must contain all the variables in the problem.

    Parameters
    ----------
    prob : Problem
        The problem to partially optimize.
    opt_vars : list, optional
        The variables to optimize over.
    dont_opt_vars : list, optional
        The variables to not optimize over.

    Returns
    -------
    Expression
        An expression representing the partial optimization.
    """
    # One of the two arguments must be specified.
    if opt_vars is None and dont_opt_vars is None:
        raise ValueError(
            "partial_optimize called with neither opt_vars nor dont_opt_vars."
        )
    # If opt_vars is not specified, it's the complement of dont_opt_vars.
    elif opt_vars is None:
        ids = [id(var) for var in dont_opt_vars]
        opt_vars = [var for var in prob.variables() if not id(var) in ids]
    # If dont_opt_vars is not specified, it's the complement of opt_vars.
    elif dont_opt_vars is None:
        ids = [id(var) for var in opt_vars]
        dont_opt_vars = [var for var in prob.variables() if not id(var) in ids]
    elif opt_vars is not None and dont_opt_vars is not None:
        ids = [id(var) for var in opt_vars + dont_opt_vars]
        for var in prob.variables():
            if id(var) not in ids:
                raise ValueError(
                    ("If opt_vars and new_opt_vars are both specified, "
                     "they must contain all variables in the problem.")
                )

    return PartialProblem(prob, opt_vars, dont_opt_vars)

class PartialProblem(Expression):
    """A partial optimization problem.

    Attributes
    ----------
    opt_vars : list
        The variables to optimize over.
    dont_opt_vars : list
        The variables to not optimize over.
    """
    def __init__(self, prob, opt_vars, dont_opt_vars):
        self.opt_vars = opt_vars
        self.dont_opt_vars = dont_opt_vars
        self.args = [prob]
        self.init_dcp_attr()
        super(PartialProblem, self).__init__()

    def init_dcp_attr(self):
        """Determines the curvature, sign, and shape from the arguments.
        """
        sign = self.args[0].objective.args[0]._dcp_attr.sign
        if isinstance(self.args[0].objective, Minimize):
            curvature = u.curvature.Curvature.CONVEX
        elif isinstance(self.args[0].objective, Maximize):
            curvature = u.curvature.Curvature.CONCAVE
        else:
            raise Exception(
                ("You called partial_optimize with a Problem object that "
                 "contains neither a Minimize nor a Maximize statement; "
                 "this is not supported.")
            )
        self._dcp_attr = u.DCPAttr(sign,
                                   curvature,
                                   u.Shape(1, 1))

    def name(self):
        """Returns the string representation of the expression.
        """
        return "PartialProblem(%s)" % str(self.args[0])

    def variables(self):
        """Returns the variables in the problem.
        """
        return copy.copy(self.dont_opt_vars)

    def parameters(self):
        """Returns the parameters in the problem.
        """
        return self.args[0].parameters()

    @property
    def value(self):
        """Returns the numeric value of the expression.

        Returns:
            A numpy matrix or a scalar.
        """
        old_vals = {var.id:var.value for var in self.variables()}
        fix_vars = []
        for var in self.variables():
            if var.value is None:
                return None
            else:
                fix_vars += [var == var.value]
        prob = Problem(type(self.args[0].objective)(self), fix_vars)
        result = prob.solve()
        # Restore the original values to the variables.
        for var in self.variables():
            var.value = old_vals[var.id]
        return result

    @staticmethod
    def _replace_new_vars(expr, id_to_new_var):
        """Replaces the given variables in the expression.

        Parameters
        ----------
        expr : LinOp
            The expression to replace variables in.
        id_to_new_var : dict
            A map of id to new variable.

        Returns
        -------
        LinOp
            An LinOp identical to expr, but with the given variables replaced.
        """
        if expr.type == lo.VARIABLE and expr.data in id_to_new_var:
            return id_to_new_var[expr.data]
        else:
            new_args = []
            for arg in expr.args:
                new_args.append(
                    PartialProblem._replace_new_vars(arg, id_to_new_var)
                )
            return lo.LinOp(expr.type, expr.size, new_args, expr.data)

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Change the ids of all the opt_vars.

        Returns
        -------
            A tuple of (affine expression, [constraints]).
        """
        id_to_new_var = {v.id:lu.create_var(v.size) for v in self.opt_vars}
        obj, constr = self.args[0].canonical_form
        obj = self._replace_new_vars(obj, id_to_new_var)
        new_constr = []
        for con in constr:
            expr = self._replace_new_vars(con.expr, id_to_new_var)
            new_constr += [type(con)(expr, con.constr_id, con.size)]
        return obj, new_constr
