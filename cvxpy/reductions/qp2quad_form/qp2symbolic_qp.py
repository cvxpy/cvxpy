"""
Copyright 2017 Robin Verschueren

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

from cvxpy.problems.problem import Problem
from cvxpy.reductions.canonicalize import canonicalize_constr, canonicalize_tree
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

from .atom_canonicalizers import CANON_METHODS as qp_canon_methods


class Qp2SymbolicQp(Reduction):
    """
    Reduces a quadratic problem to a problem that consists of affine expressions
    and symbolic quadratic forms.
    """

    def accepts(self, problem):
        return problem.is_qp()

    def apply(self, problem):
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to symbolic QP")
        inverse_data = InverseData(problem)

        new_obj, new_constrs = canonicalize_tree(problem.objective, qp_canon_methods)
        for constr in problem.constraints:
            top_constr, canon_constrs = canonicalize_constr(constr, qp_canon_methods)
            new_constrs += canon_constrs + [top_constr]
            inverse_data.cons_id_map[constr.id] = top_constr.id

        new_problem = Problem(new_obj, new_constrs)
        return new_problem, [inverse_data]

    def invert(self, solution, inverse_data):
        inv = inverse_data.pop()

        pvars = {id: solution.primal_vars[id] for id in inv.id_map if id in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[id] for orig_id, id in inv.cons_id_map.items()}

        return Solution(solution.status, solution.opt_val, pvars, dvars)
