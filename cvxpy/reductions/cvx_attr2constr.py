
"""
Copyright 2017 Steven Diamond

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

from cvxpy.reductions import Reduction, Solution
from cvxpy.atoms import diag, reshape
from cvxpy.expressions.constants import Constant
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.variable import Variable, upper_tri_to_full
import numpy as np
import scipy.sparse as sp


# Convex attributes that generate constraints.
CONVEX_ATTRIBUTES = [
    'nonneg',
    'nonpos',
    'symmetric',
    'diag',
    'PSD',
    'NSD',
]

# Attributes related to symmetry.
SYMMETRIC_ATTRIBUTES = [
    'symmetric',
    'PSD',
    'NSD',
]


def convex_attributes(variables):
    """Returns a list of the (constraint-generating) convex attributes present
       among the variables.
    """
    return attributes_present(variables, CONVEX_ATTRIBUTES)


def attributes_present(variables, attr_map):
    """Returns a list of the relevant attributes present
       among the variables.
    """
    return [attr for attr in attr_map if any(v.attributes[attr] for v
                                             in variables)]


class CvxAttr2Constr(Reduction):
    """Expand convex variable attributes into constraints."""

    def accepts(self, problem):
        return True

    def apply(self, problem):
        if not attributes_present(problem.variables(), CONVEX_ATTRIBUTES):
            return problem, ()

        # For each unique variable, add constraints.
        id2new_var = {}
        id2new_obj = {}
        id2old_var = {}
        constr = []
        for var in problem.variables():
            if var.id not in id2new_var:
                id2old_var[var.id] = var
                new_var = False
                new_attr = var.attributes.copy()
                for key in CONVEX_ATTRIBUTES:
                    if new_attr[key]:
                        new_var = True
                        new_attr[key] = False

                if attributes_present([var], SYMMETRIC_ATTRIBUTES):
                    n = var.shape[0]
                    shape = (n*(n+1)//2, 1)
                    upper_tri = Variable(shape, **new_attr)
                    id2new_var[var.id] = upper_tri
                    fill_coeff = Constant(upper_tri_to_full(n))
                    full_mat = fill_coeff*upper_tri
                    obj = reshape(full_mat, (n, n))
                elif var.attributes['diag']:
                    diag_var = Variable(var.shape[0], **new_attr)
                    id2new_var[var.id] = diag_var
                    obj = diag(diag_var)
                elif new_var:
                    obj = Variable(var.shape, **new_attr)
                    id2new_var[var.id] = obj
                else:
                    obj = var
                    id2new_var[var.id] = obj

                id2new_obj[id(var)] = obj
                if var.is_nonneg():
                    constr.append(obj >= 0)
                elif var.is_nonpos():
                    constr.append(obj <= 0)
                elif var.is_psd():
                    constr.append(obj >> 0)
                elif var.attributes['NSD']:
                    constr.append(obj << 0)

        # Create new problem.
        obj = problem.objective.tree_copy(id_objects=id2new_obj)
        cons_id_map = {}
        for cons in problem.constraints:
            constr.append(cons.tree_copy(id_objects=id2new_obj))
            cons_id_map[cons.id] = constr[-1].id
        inverse_data = (id2new_var, id2old_var, cons_id_map)
        return cvxtypes.problem()(obj, constr), inverse_data

    def invert(self, solution, inverse_data):
        if not inverse_data:
            return solution

        id2new_var, id2old_var, cons_id_map = inverse_data
        pvars = {}
        for id, var in id2old_var.items():
            new_var = id2new_var[id]
            # Need to map from constrained to symmetric variable.
            if new_var.id in solution.primal_vars:
                if var.attributes['diag']:
                    pvars[id] = sp.diags(solution.primal_vars[new_var.id].flatten(), [0])
                elif attributes_present([var], SYMMETRIC_ATTRIBUTES):
                    n = var.shape[0]
                    value = np.zeros(var.shape)
                    idxs = np.triu_indices(n)
                    value[idxs] = solution.primal_vars[new_var.id].flatten()
                    value += value.T - np.diag(value.diagonal())
                    pvars[id] = value
                else:
                    pvars[id] = var.project(solution.primal_vars[new_var.id])

        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in cons_id_map.items()
                 if vid in solution.dual_vars}
        return Solution(solution.status, solution.opt_val, pvars,
                        dvars, solution.attr)
