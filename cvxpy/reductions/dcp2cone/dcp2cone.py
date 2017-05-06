from cvxpy.reductions.reduction import Reduction
from ..canonicalize import canonicalize_constr, canonicalize_expr, canonicalize_tree
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS
import cvxpy


class Dcp2Cone(Reduction):
    def __init__(self):
        self.old_var_ids = dict() # A list of the old variable IDs.
        self.constr_map = dict()  # Maps the new constraint IDs to the old constraint IDs.

    def accepts(self, prob):
        return prob.is_dcp()

    def apply(self, prob):
        self.old_var_ids = [v.id for v in prob.variables()]

        obj_expr, new_constrs = canonicalize_tree(prob.objective.args[0],
                                            canon_methods=CANON_METHODS)
        if isinstance(prob.objective, cvxpy.Minimize):
           new_obj = cvxpy.Minimize(obj_expr)
        elif isinstance(prob.objective, cvxpy.Maximize):
           new_obj = cvxpy.Maximize(obj_expr)

        for c in prob.constraints:
            top_constr, canon_constrs = canonicalize_constr(c,
                                                canon_methods=CANON_METHODS)
            #print(canon_constrs)
            new_constrs += canon_constrs + [top_constr]
            self.constr_map.update({ top_constr.id : c.id })

        new_prob = cvxpy.Problem(new_obj, new_constrs)
        return new_prob


    #@staticmethod
    #def canonicalize_expr(expr):
    #    canon_args = []
    #    for arg in expr.args:
    #        canon_args += [canonicalize_expr]
    #    canon_expr = CANON[type(expr)](expr)
    #    return canon_expr




    def invert(self, solution, inverse_data):

        pvars = dict()
        for id, val in solution.primal_vars.items():
            if id in self.old_var_ids:
                pvars.update({id: val})

        for old_id, orig_id in self.constr_map.items:
            orig_sol.update({orig_id : solution.dual_vars[old_id]})

        orig_sol.optval = old_sol.optval
        orig_sol.status = old_sol.status

        orig_sol = Solution(status, opt_val, primal_vars, dual_vars, attr)

        return orig_sol
