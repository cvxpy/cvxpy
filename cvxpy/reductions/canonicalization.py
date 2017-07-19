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
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
from cvxpy.reductions import InverseData, Reduction, Solution


class Canonicalization(Reduction):

    def __init__(self, canon_methods=None):
        self.canon_methods = canon_methods

    def apply(self, problem):
        inverse_data = InverseData(problem)

        new_objective, new_constraints = self.canonicalize_tree(problem.objective)
        for constraint in problem.constraints:
            constraint_copy, canon_constraints = self.canonicalize_tree(constraint)
            new_constraints += canon_constraints + [constraint_copy]
            inverse_data.cons_id_map.update({constraint.id: constraint_copy.id})

        new_problem = Problem(new_objective, new_constraints)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        pvars = {id: solution.primal_vars[id] for id in inverse_data.id_map
                 if id in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[id]
                 for orig_id, id in inverse_data.cons_id_map.items()}
        return Solution(solution.status, solution.opt_val, pvars, dvars)

    def canonicalize_tree(self, expr):
        canon_args = []
        constrs = []
        for arg in expr.args:
            canon_arg, c = self.canonicalize_tree(arg)
            canon_args += [canon_arg]
            constrs += c
        canon_expr, c = self.canonicalize_expr(expr, canon_args)
        constrs += c
        return canon_expr, constrs

    def canonicalize_expr(self, expr, args):
        if type(expr) in self.canon_methods:
            return self.canon_methods[type(expr)](expr, args)
        else:
            return expr.copy(args), []
