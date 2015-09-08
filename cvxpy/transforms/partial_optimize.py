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
        # Replace the opt_vars in prob with new variables.
        id_to_new_var = {var.id:var.copy() for var in self.opt_vars}
        new_obj = self._replace_new_vars(prob.objective, id_to_new_var)
        new_constrs = [self._replace_new_vars(con, id_to_new_var)
                       for con in prob.constraints]
        self._prob = Problem(new_obj, new_constrs)
        self.init_dcp_attr()
        super(PartialProblem, self).__init__()

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.opt_vars, self.dont_opt_vars]

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
        prob = Problem(self.args[0].objective.copy(self), fix_vars)
        result = prob.solve()
        # Restore the original values to the variables.
        for var in self.variables():
            var.value = old_vals[var.id]
        return result

    @staticmethod
    def _replace_new_vars(obj, id_to_new_var):
        """Replaces the given variables in the object.

        Parameters
        ----------
        obj : Object
            The object to replace variables in.
        id_to_new_var : dict
            A map of id to new variable.

        Returns
        -------
        Object
            An object identical to obj, but with the given variables replaced.
        """
        if isinstance(obj, Variable) and obj.id in id_to_new_var:
            return id_to_new_var[obj.id]
        # Leaves outside of optimized variables are preserved.
        elif len(obj.args) == 0:
            return obj
        elif isinstance(obj, PartialProblem):
            prob = obj.args[0]
            new_obj = PartialProblem._replace_new_vars(prob.objective,
                id_to_new_var)
            new_constr = []
            for constr in prob.constraints:
                new_constr.append(
                    PartialProblem._replace_new_vars(constr,
                                            id_to_new_var)
                )
            new_args = [Problem(new_obj, new_constr)]
            return obj.copy(new_args)
        # Parent nodes are copied.
        else:
            new_args = []
            for arg in obj.args:
                new_args.append(
                    PartialProblem._replace_new_vars(arg, id_to_new_var)
                )
            return obj.copy(new_args)

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Change the ids of all the opt_vars.

        Returns
        -------
            A tuple of (affine expression, [constraints]).
        """
        return self._prob.canonical_form
