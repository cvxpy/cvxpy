"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal, 2017 Robin Verschueren

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

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.canonicalize import canonicalize_tree
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution


class Canonicalization(Reduction):

    def __init__(self, canon_methods=None):
        self.canon_methods = canon_methods

    def accepts(self, problem):
        raise NotImplementedError

    def apply(self, problem):
        inverse_data = InverseData(problem)

        new_obj, new_constrs = canonicalize_tree(problem.objective, self.canon_methods)
        for con in problem.constraints:
            top_constr, canon_constrs = canonicalize_tree(con, self.canon_methods)
            new_constrs += canon_constrs + [top_constr]
            inverse_data.cons_id_map.update({con.id: top_constr.id})

        new_problem = Problem(new_obj, new_constrs)
        return new_problem, [inverse_data]

    def invert(self, solution, inverse_data):
        inv = inverse_data.pop()
        pvars = {id: solution.primal_vars[id] for id in inv.id_map if id in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[id] for orig_id, id in inv.cons_id_map.items()}
        return Solution(solution.status, solution.opt_val, pvars, dvars)

    def canonicalize_tree(self, expr, canon_methods):
        canon_args = []
        constrs = []
        for arg in expr.args:
            canon_arg, c = canonicalize_tree(arg, canon_methods)
            canon_args += [canon_arg]
            constrs += c
        canon_expr, c = self.canonicalize_expr(expr, canon_args, canon_methods)
        constrs += c
        return canon_expr, constrs

    def canonicalize_expr(self, expr, args, canon_methods):
        if isinstance(expr, Minimize):
            return Minimize(*args), []
        elif isinstance(expr, Maximize):
            return Maximize(*args), []
        elif isinstance(expr, Variable):
            return expr, []
        elif isinstance(expr, Constant):
            return expr, []
        elif isinstance(expr, Constraint):
            return type(expr)(*args), []
        elif expr.is_atom_convex() and expr.is_atom_concave():
            if isinstance(expr, AddExpression):
                expr = type(expr)(args)
            else:
                expr = type(expr)(*args)
            return expr, []
        else:
            return canon_methods[type(expr)](expr, args)
