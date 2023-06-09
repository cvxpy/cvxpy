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

from typing import List, Optional, Tuple

import cvxpy.settings as s
import cvxpy.utilities as u
from cvxpy.atoms import sum, trace
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem


def partial_optimize(
    prob: Problem,
    opt_vars: Optional[List[Variable]] = None,
    dont_opt_vars: Optional[List[Variable]] = None,
    solver=None,
    **kwargs
) -> "PartialProblem":
    """Partially optimizes the given problem over the specified variables.

    Either opt_vars or dont_opt_vars must be given.
    If both are given, they must contain all the variables in the problem.

    Partial optimize is useful for two-stage optimization and graph implementations.
    For example, we can write

    .. code :: python

        x = Variable(n)
        t = Variable(n)
        abs_x = partial_optimize(Problem(Minimize(sum(t)),
                  [-t <= x, x <= t]), opt_vars=[t])

    to define the entrywise absolute value of x.

    Parameters
    ----------
    prob : Problem
        The problem to partially optimize.
    opt_vars : list, optional
        The variables to optimize over.
    dont_opt_vars : list, optional
        The variables to not optimize over.
    solver : str, optional
        The default solver to use for value and grad.
    kwargs : keywords, optional
        Additional solver specific keyword arguments.

    Returns
    -------
    Expression
        An expression representing the partial optimization.
        Convex for minimization objectives and concave for maximization objectives.
    """
    # One of the two arguments must be specified.
    if opt_vars is None and dont_opt_vars is None:
        raise ValueError(
            "partial_optimize called with neither opt_vars nor dont_opt_vars."
        )
    # If opt_vars is not specified, it's the complement of dont_opt_vars.
    elif opt_vars is None:
        ids = [id(var) for var in dont_opt_vars]
        opt_vars = [var for var in prob.variables() if id(var) not in ids]
    # If dont_opt_vars is not specified, it's the complement of opt_vars.
    elif dont_opt_vars is None:
        ids = [id(var) for var in opt_vars]
        dont_opt_vars = [var for var in prob.variables() if id(var) not in ids]
    elif opt_vars is not None and dont_opt_vars is not None:
        ids = [id(var) for var in opt_vars + dont_opt_vars]
        for var in prob.variables():
            if id(var) not in ids:
                raise ValueError(
                    ("If opt_vars and new_opt_vars are both specified, "
                     "they must contain all variables in the problem.")
                )

    # Replace the opt_vars in prob with new variables.
    id_to_new_var = {id(var): Variable(var.shape,
                                       **var.attributes) for var in opt_vars}
    new_obj = prob.objective.tree_copy(id_to_new_var)
    new_constrs = [con.tree_copy(id_to_new_var)
                   for con in prob.constraints]
    new_var_prob = Problem(new_obj, new_constrs)
    return PartialProblem(new_var_prob, opt_vars, dont_opt_vars, solver, **kwargs)


class PartialProblem(Expression):
    """A partial optimization problem.

    Attributes
    ----------
    opt_vars : list
        The variables to optimize over.
    dont_opt_vars : list
        The variables to not optimize over.
    """

    def __init__(
            self, prob: Problem, opt_vars: List[Variable],
            dont_opt_vars: List[Variable], solver, **kwargs) -> None:
        self.opt_vars = opt_vars
        self.dont_opt_vars = dont_opt_vars
        self.solver = solver
        self.args = [prob]
        self._solve_kwargs = kwargs
        super(PartialProblem, self).__init__()

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.opt_vars, self.dont_opt_vars, self.solver]

    def is_constant(self) -> bool:
        return len(self.args[0].variables()) == 0

    def is_convex(self) -> bool:
        """Is the expression convex?
        """
        return self.args[0].is_dcp() and \
            type(self.args[0].objective) == Minimize

    def is_concave(self) -> bool:
        """Is the expression concave?
        """
        return self.args[0].is_dcp() and \
            type(self.args[0].objective) == Maximize

    def is_dpp(self, context: str = 'dcp') -> bool:
        """The expression is a disciplined parameterized expression.
        """
        if context.lower() in ['dcp', 'dgp']:
            return self.args[0].is_dpp(context)
        else:
            raise ValueError("Unsupported context", context)

    def is_log_log_convex(self) -> bool:
        """Is the expression convex?
        """
        return self.args[0].is_dgp() and \
            type(self.args[0].objective) == Minimize

    def is_log_log_concave(self) -> bool:
        """Is the expression convex?
        """
        return self.args[0].is_dgp() and \
            type(self.args[0].objective) == Maximize

    def is_nonneg(self) -> bool:
        """Is the expression nonnegative?
        """
        return self.args[0].objective.args[0].is_nonneg()

    def is_nonpos(self) -> bool:
        """Is the expression nonpositive?
        """
        return self.args[0].objective.args[0].is_nonpos()

    def is_imag(self) -> bool:
        """Is the Leaf imaginary?
        """
        return False

    def is_complex(self) -> bool:
        """Is the Leaf complex valued?
        """
        return False

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the (row, col) dimensions of the expression.
        """
        return tuple()

    def name(self) -> str:
        """Returns the string representation of the expression.
        """
        return f"PartialProblem({self.args[0]})"

    def variables(self) -> List[Variable]:
        """Returns the variables in the problem.
        """
        return self.args[0].variables()

    def parameters(self):
        """Returns the parameters in the problem.
        """
        return self.args[0].parameters()

    def constants(self) -> List[Constant]:
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
        for var in self.dont_opt_vars:
            if var.value is None:
                return u.grad.error_grad(self)
            else:
                fix_vars += [var == var.value]
        prob = Problem(self.args[0].objective,
                       fix_vars + self.args[0].constraints)
        prob.solve(solver=self.solver, **self._solve_kwargs)
        # Compute gradient.
        if prob.status in s.SOLUTION_PRESENT:
            sign = self.is_convex() - self.is_concave()
            # Form Lagrangian.
            lagr = self.args[0].objective.args[0]
            for constr in self.args[0].constraints:
                # TODO: better way to get constraint expressions.
                lagr_multiplier = self.cast_to_const(sign * constr.dual_value)
                prod = lagr_multiplier.T @ constr.expr
                if prod.is_scalar():
                    lagr += sum(prod)
                else:
                    lagr += trace(prod)
            grad_map = lagr.grad
            result = {var: grad_map[var] for var in self.dont_opt_vars}
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
        # Variables optimized over are replaced in self.args[0].
        obj_expr = self.args[0].objective.args[0]
        return self.args[0].constraints + obj_expr.domain

    @property
    def value(self):
        """Returns the numeric value of the expression.

        Returns:
            A numpy matrix or a scalar.
        """
        old_vals = {var.id: var.value for var in self.variables()}
        fix_vars = []
        for var in self.dont_opt_vars:
            if var.value is None:
                return None
            else:
                fix_vars += [var == var.value]
        prob = Problem(self.args[0].objective, fix_vars + self.args[0].constraints)
        prob.solve(solver=self.solver, **self._solve_kwargs)
        # Restore the original values to the variables.
        for var in self.variables():
            var.value = old_vals[var.id]
        # Need to get value returned by solver
        # in case of stacked partial_optimizes.
        return prob._solution.opt_val

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Change the ids of all the opt_vars.

        Returns
        -------
            A tuple of (affine expression, [constraints]).
        """
        # Canonical form for objective and problem switches from minimize
        # to maximize.
        obj, constrs = self.args[0].objective.args[0].canonical_form
        for cons in self.args[0].constraints:
            constrs += cons.canonical_form[1]
        return (obj, constrs)
