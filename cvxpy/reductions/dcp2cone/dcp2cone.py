from cvxpy.reductions.reduction import Reduction
from ..canonicalize import canonicalize_constr, canonicalize_tree
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS
from cvxpy.reductions.inverse_data import InverseData
import cvxpy


class Dcp2Cone(Reduction):

    def accepts(self, problem):
        return problem.is_dcp()

    def apply(self, problem):
        inverse_data = InverseData(problem)

        obj_expr, new_constrs = canonicalize_tree(
            problem.objective.args[0],
            canon_methods=CANON_METHODS
        )
        if isinstance(problem.objective, cvxpy.Minimize):
            new_obj = cvxpy.Minimize(obj_expr)
        elif isinstance(problem.objective, cvxpy.Maximize):
            new_obj = cvxpy.Maximize(obj_expr)

        for c in problem.constraints:
            top_constr, canon_constrs = canonicalize_constr(
                c,
                canon_methods=CANON_METHODS
            )
            new_constrs += canon_constrs + [top_constr]
            inverse_data.cons_id_map.update({top_constr.id: c.id})

        new_problem = cvxpy.Problem(new_obj, new_constrs)
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
