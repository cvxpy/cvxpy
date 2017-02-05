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
import cvxpy.utilities as u
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.problems.problem import Problem
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.expression import Expression
from cvxpy.atoms import trace
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
        id_to_new_var = {var.id: var.copy() for var in self.opt_vars}
        new_obj = self._replace_new_vars(prob.objective, id_to_new_var)
        new_constrs = [self._replace_new_vars(con, id_to_new_var)
                       for con in prob.constraints]
        self._prob = Problem(new_obj, new_constrs)
        super(PartialProblem, self).__init__()

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.opt_vars, self.dont_opt_vars]

    def is_convex(self):
        """Is the expression convex?
        """
        return self.args[0].is_dcp() and \
            type(self.args[0].objective) == Minimize

    def is_concave(self):
        """Is the expression concave?
        """
        return self.args[0].is_dcp() and \
            type(self.args[0].objective) == Maximize

    def is_positive(self):
        """Is the expression positive?
        """
        return self.args[0].objective.args[0].is_positive()

    def is_negative(self):
        """Is the expression negative?
        """
        return self.args[0].objective.args[0].is_negative()

    @property
    def size(self):
        """Returns the (row, col) dimensions of the expression.
        """
        return (1, 1)

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

    def constants(self):
        """Returns the constants in the problem.
        """
        return self.args[0].constants()

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.
        None indicates variable values unknown or outside domain.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        # Subgrad of g(y) = min f_0(x,y)
        #                   s.t. f_i(x,y) <= 0, i = 1,..,p
        #                        h_i(x,y) == 0, i = 1,...,q
        # Given by Df_0(x^*,y) + \sum_i Df_i(x^*,y) \lambda^*_i
        #          + \sum_i Dh_i(x^*,y) \nu^*_i
        # where x^*, \lambda^*_i, \nu^*_i are optimal primal/dual variables.
        # Add PSD constraints in same way.

        # Short circuit for constant.
        if self.is_constant():
            return u.grad.constant_grad(self)

        old_vals = {var.id: var.value for var in self.variables()}
        fix_vars = []
        for var in self.variables():
            if var.value is None:
                return u.grad.error_grad(self)
            else:
                fix_vars += [var == var.value]
        prob = Problem(self._prob.objective,
                       fix_vars + self._prob.constraints)
        prob.solve()
        # Compute gradient.
        if prob.status in s.SOLUTION_PRESENT:
            sign = self.is_convex() - self.is_concave()
            # Form Lagrangian.
            lagr = self._prob.objective.args[0]
            for constr in self._prob.constraints:
                # TODO: better way to get constraint expressions.
                lagr_multiplier = self.cast_to_const(sign*constr.dual_value)
                lagr += trace(lagr_multiplier.T*constr._expr)
            grad_map = lagr.grad
            result = {var: grad_map[var] for var in self.variables()}
        else:  # Unbounded, infeasible, or solver error.
            result = u.grad.error_grad(self)
        # Restore the original values to the variables.
        for var in self.variables():
            var.value = old_vals[var.id]
        return result

    @property
    def domain(self):
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        # Variables optimized over are replaced in self._prob.
        obj_expr = self._prob.objective.args[0]
        return self._prob.constraints + obj_expr.domain

    @property
    def value(self):
        """Returns the numeric value of the expression.

        Returns:
            A numpy matrix or a scalar.
        """
        old_vals = {var.id: var.value for var in self.variables()}
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
        # Canonical form for objective and problem switches from minimize
        # to maximize.
        obj, constrs = self._prob.objective.args[0].canonical_form
        for cons in self._prob.constraints:
            constrs += cons.canonical_form[1]
        return (obj, constrs)
