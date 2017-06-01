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

from cvxpy.problems.problem import Problem
from cvxpy.reductions.canonicalize import canonicalize_tree
from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS as cone_canon_methods
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution


class Dcp2Cone(Reduction):

    def accepts(self, problem):
        return problem.is_dcp()

    def apply(self, problem):
        inverse_data = InverseData(problem)

        new_obj, new_constrs = canonicalize_tree(problem.objective, cone_canon_methods)
        for con in problem.constraints:
            top_constr, canon_constrs = canonicalize_tree(con, cone_canon_methods)
            new_constrs += canon_constrs + [top_constr]
            inverse_data.cons_id_map.update({top_constr.id: con.id})

        new_problem = Problem(new_obj, new_constrs)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):

        orig_sol = solution
        pvars = dict()
        for id, val in solution.primal_vars.items():
            if id in inverse_data.id_map.keys():
                pvars.update({id: val})

        dvars = dict()
        for old_id, orig_id in inverse_data.cons_id_map.items():
            dvars.update({orig_id: solution.dual_vars[old_id]})

        opt_val = solution.optval
        status = solution.status

        orig_sol = Solution(status, opt_val, pvars, dvars, {})

        return orig_sol
