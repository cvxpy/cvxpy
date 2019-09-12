"""
Copyright 2017 Steven Diamond

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

from cvxpy.reductions import Reduction, Solution
from cvxpy.atoms import diag, reshape
from cvxpy.expressions.constants import Constant
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.variable import Variable, upper_tri_to_full
from cvxpy.reductions import InverseData
from cvxpy.reductions.utilities import tensor_mul
import numpy as np
import scipy.sparse as sp


# Convex attributes that generate constraints.
CONVEX_ATTRIBUTES = [
    'nonneg',
    'nonpos',
    'pos',
    'neg',
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

        inverse_data = InverseData(problem)
        # For each unique variable, add constraints.
        id2new_var = {}
        id2new_obj = {}
        id2old_var = {}
        constr = []
        primal_tensor = {}
        for var in problem.variables():
            if var.id not in id2new_var:
                id2old_var[var.id] = var
                new_var = False
                new_attr = var.attributes.copy()
                for key in CONVEX_ATTRIBUTES:
                    if new_attr[key]:
                        new_var = True
                        new_attr[key] = False

                # The mapping from new variable value to old variable value.
                coeff_mat = None
                if attributes_present([var], SYMMETRIC_ATTRIBUTES):
                    n = var.shape[0]
                    shape = (n*(n+1)//2, 1)
                    upper_tri = Variable(shape, **new_attr)
                    id2new_var[var.id] = upper_tri
                    fill_coeff = upper_tri_to_full(n)
                    full_mat = Constant(fill_coeff)*upper_tri
                    obj = reshape(full_mat, (n, n))
                    coeff_mat = fill_coeff
                elif var.attributes['diag']:
                    diag_var = Variable(var.shape[0], **new_attr)
                    id2new_var[var.id] = diag_var
                    obj = diag(diag_var)
                    # Column j is offset j*(n + 1)
                    vals = np.ones(var.shape[0])
                    rows = np.arange(diag_var.size, var.shape[0] + 1)
                    cols = np.arange(var.shape[0])
                    mat = sp.coo_matrix((vals, (rows, cols)),
                                        shape=(diag_var.size, var.shape[0]))
                    coeff_mat = mat.tocsc()
                elif new_var:
                    obj = Variable(var.shape, **new_attr)
                    id2new_var[var.id] = obj
                else:
                    obj = var
                    id2new_var[var.id] = obj

                # Map from new to old variable value.
                if coeff_mat is None:
                    # Default is identity.
                    coeff_mat = sp.eye(var.size, var.size, format="csc")
                primal_tensor[var.id] = {id2new_var[var.id].id: coeff_mat}

                id2new_obj[id(var)] = obj
                if var.is_pos() or var.is_nonneg():
                    constr.append(obj >= 0)
                elif var.is_neg() or var.is_nonpos():
                    constr.append(obj <= 0)
                elif var.is_psd():
                    constr.append(obj >> 0)
                elif var.attributes['NSD']:
                    constr.append(obj << 0)
        inverse_data.primal_tensor = primal_tensor

        # Create new problem.
        obj = problem.objective.tree_copy(id_objects=id2new_obj)
        dual_tensor = {}
        for cons in problem.constraints:
            constr.append(cons.tree_copy(id_objects=id2new_obj))
            for dv_old, dv_new in zip(cons.dual_variables,
                                      constr[-1].dual_variables):
                dual_tensor[dv_old.id] = {dv_new.id: sp.eye(dv_new.size,
                                                            dv_new.size,
                                                            format="csc")}
        inverse_data.dual_tensor = dual_tensor
        return cvxtypes.problem()(obj, constr), inverse_data

    def invert(self, solution, inverse_data):
        if not inverse_data:
            return solution
        pvars = tensor_mul(inverse_data.primal_tensor, solution.primal_vars)
        dvars = tensor_mul(inverse_data.dual_tensor, solution.dual_vars)

        return Solution(solution.status, solution.opt_val, pvars,
                        dvars, solution.attr)
