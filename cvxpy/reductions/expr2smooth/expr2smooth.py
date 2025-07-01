"""
Copyright 2025 CVXPY developers

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

from typing import Tuple

import cvxpy as cp
from cvxpy import problems
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.expr2smooth.canonicalizers import CANON_METHODS as smooth_canon_methods
from cvxpy.reductions.inverse_data import InverseData


class Expr2smooth(Canonicalization):
    """Reduce Expressions to an equivalent smooth program

    This reduction takes as input (minimization) expressions and converts
    them into smooth expressions.
    """
    def __init__(self, problem=None, quad_obj: bool = False) -> None:
        super(Canonicalization, self).__init__(problem=problem)
        self.smooth_canon_methods = smooth_canon_methods
        self.quad_obj = quad_obj

    def accepts(self, problem):
        """A problem is always accepted"""
        return True

    def apply(self, problem):
        """Converts an expr to a smooth program"""
        inverse_data = InverseData(problem)

        # smoothen objective function
        canon_objective, canon_constraints = self.canonicalize_tree(
            problem.objective, True)

        # smoothen constraints
        for constraint in problem.constraints:
            # canon_constr is the constraint re-expressed in terms of
            # its canonicalized arguments, and aux_constr are the constraints
            # generated while canonicalizing the arguments of the original
            # constraint
            canon_constr, aux_constr = self.canonicalize_tree(
                constraint, False)
            canon_constraints += aux_constr + [canon_constr]
            inverse_data.cons_id_map.update({constraint.id: canon_constr.id})

        new_problem = problems.problem.Problem(canon_objective,
                                               canon_constraints)
        return new_problem, inverse_data

    def canonicalize_tree(self, expr, affine_above: bool) -> Tuple[Expression, list]:
        """Recursively canonicalize an Expression.

        Parameters
        ----------
        expr : The expression tree to canonicalize.
        affine_above : The path up to the root node is all affine atoms.

        Returns
        -------
        A tuple of the canonicalized expression and generated constraints.
        """
        # TODO don't copy affine expressions?
        affine_atom = type(expr) not in self.smooth_canon_methods
        canon_args = []
        constrs = []
        for arg in expr.args:
            canon_arg, c = self.canonicalize_tree(arg, affine_atom and affine_above)
            canon_args += [canon_arg]
            constrs += c
        canon_expr, c = self.canonicalize_expr(expr, canon_args, affine_above)
        constrs += c
        return canon_expr, constrs

    def canonicalize_expr(self, expr, args, affine_above: bool) -> Tuple[Expression, list]:
        """Canonicalize an expression, w.r.t. canonicalized arguments.

        Parameters
        ----------
        expr : The expression tree to canonicalize.
        args : The canonicalized arguments of expr.
        affine_above : The path up to the root node is all affine atoms.

        Returns
        -------
        A tuple of the canonicalized expression and generated constraints.
        """
        # Constant trees are collapsed, but parameter trees are preserved.
        if isinstance(expr, Expression) and (
                expr.is_constant() and not expr.parameters()):
            return expr, []

        if type(expr) in self.smooth_canon_methods:
            return self.smooth_canon_methods[type(expr)](expr, args)

        return expr.copy(args), []

"""
def example_max():
    # Define variables
    x = cp.Variable(1)
    y = cp.Variable(1)
    
    objective = cp.Minimize(-cp.maximum(x,y))
    
    constraints = [x - 14 == 0, y - 6 == 0]
    
    problem = cp.Problem(objective, constraints)
    return problem

def example_sqrt():
    # Define variables
    x = cp.Variable(1)
    
    objective = cp.Minimize(cp.sqrt(x))
    
    constraints = [x - 4 == 0]
    
    problem = cp.Problem(objective, constraints)
    return problem

def example_pnorm_even():
    # Define variables
    x = cp.Variable(2)
    
    objective = cp.Minimize(cp.pnorm(x, p=2))
    
    constraints = [x[0] - 3 == 0, x[1] - 4 == 0]
    
    problem = cp.Problem(objective, constraints)
    return problem

def example_pnorm_odd():
    # Define variables
    x = cp.Variable(2)
    
    objective = cp.Minimize(cp.pnorm(x, p=3))
    
    constraints = [x[0] - 3 == 0, x[1] - 4 == 0]
    
    problem = cp.Problem(objective, constraints)
    return problem

prob = example_sqrt()
new_problem, inverse = Expr2smooth(prob).apply(prob)
print(new_problem)
"""