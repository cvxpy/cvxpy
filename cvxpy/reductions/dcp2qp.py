from cvxpy.atoms.quad_form import QuadForm
from cvxpy.reductions.reduction import Reduction
from canonicalize import canonicalize_constr, canonicalize_tree
from cvxpy.reductions.solution import Solution

from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS
from cvxpy.atoms import *

import cvxpy
from numpy import eye

from cvxpy.expressions.variables.variable import Variable

QP_CANON_METHODS={}

def quad_over_lin_QPcanon(expr, args):
    x = args[0]
    y = args[1]
    shape = expr.shape
    # precondition: shape == (1,)
    if not (x.shape[0] == 1 or x.shape[1] == 1):
        raise ValueError("x can only be a vector in a quadratic form")
    length_x = max(x.shape[0], x.shape[1])
    return quad_form(x, eye(length_x)/y.value), []

def power_QPcanon(expr, args):
    x = args[0]
    p = expr.p
    w = expr.w

    if p != 2:
        raise ValueError("quadratic form can only have power 2")
    return NotImplemented

QP_CANON_METHODS[affine_prod] = CANON_METHODS[affine_prod]
QP_CANON_METHODS[abs] = CANON_METHODS[abs]
QP_CANON_METHODS[max_elemwise] = CANON_METHODS[max_elemwise]
QP_CANON_METHODS[sum_largest] = CANON_METHODS[sum_largest]
QP_CANON_METHODS[max_entries] = CANON_METHODS[max_entries]
QP_CANON_METHODS[pnorm] = CANON_METHODS[pnorm]

QP_CANON_METHODS[quad_over_lin] = quad_over_lin_QPcanon
QP_CANON_METHODS[power] = power_QPcanon

class Dcp2Qp(Reduction):
    def __init__(self):
        self.old_var_ids = dict() # A list of the old variable IDs.
        self.constr_map = dict()  # Maps the new constraint IDs to the old constraint IDs.

    def accepts(self, prob):
        return prob.is_qp()

    def apply(self, prob):
        self.old_var_ids = [v.id for v in prob.variables()]

        obj_expr, new_constrs = canonicalize_tree(prob.objective.args[0],
                                            canon_methods=QP_CANON_METHODS)
        if isinstance(prob.objective, cvxpy.Minimize):
           new_obj = cvxpy.Minimize(obj_expr)
        elif isinstance(prob.objective, cvxpy.Maximize):
           new_obj = cvxpy.Maximize(obj_expr)

        for c in prob.constraints:
            top_constr, canon_constrs = canonicalize_constr(c,
                                                canon_methods=QP_CANON_METHODS)
            new_constrs += canon_constrs + [top_constr]
            self.constr_map.update({ top_constr.id : c.id })

        new_prob = cvxpy.Problem(new_obj, new_constrs)
        return new_prob


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
