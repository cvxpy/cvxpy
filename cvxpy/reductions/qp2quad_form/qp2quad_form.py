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

import cvxpy
from cvxpy.reductions.canonicalize import canonicalize_constr, canonicalize_tree
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

from .atom_canonicalizers import CANON_METHODS


class Qp2QuadForm(Reduction):
    """
    Reduces a quadratic problem to a problem that consists of affine expressions
    and symbolic quadratic forms.
    """

    def accepts(self, problem):
        return problem.is_qp()

    def apply(self, problem):
        inverse_data = InverseData(problem)

        obj_expr, new_constrs = canonicalize_tree(problem.objective.args[0], CANON_METHODS)
        if isinstance(problem.objective, cvxpy.Minimize):
            new_obj = cvxpy.Minimize(obj_expr)
        elif isinstance(problem.objective, cvxpy.Maximize):
            new_obj = cvxpy.Maximize(obj_expr)

        for c in problem.constraints:
            top_constr, canon_constrs = canonicalize_constr(c, CANON_METHODS)
            new_constrs += canon_constrs + [top_constr]
            inverse_data.cons_id_map.update({top_constr.id: c.id})

        new_problem = cvxpy.Problem(new_obj, new_constrs)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        primal_vars = dict()
        dual_vars = dict()
        for id, val in solution.primal_vars.items():
            if id in inverse_data.id_map.keys():
                primal_vars.update({id: val})

        for old_id, orig_id in inverse_data.cons_id_map.items():
            dual_vars.update({orig_id: solution.dual_vars[old_id]})

        return Solution(solution.status, solution.opt_val, primal_vars, dual_vars)
