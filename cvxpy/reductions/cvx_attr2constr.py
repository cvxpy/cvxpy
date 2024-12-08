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

from typing import List

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms import diag, reshape
from cvxpy.atoms.affine.upper_tri import upper_tri_to_full
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

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
    'bounds',
    'sparsity',
]

# Attributes that define lower and upper bounds.
BOUND_ATTRIBUTES = [
    'nonneg',
    'nonpos',
    'pos',
    'neg',
    'bounds',
]

# Attributes related to symmetry.
SYMMETRIC_ATTRIBUTES = [
    'symmetric',
    'PSD',
    'NSD',
]


def convex_attributes(variables) -> list[str]:
    """Returns a list of the (constraint-generating) convex attributes present
       among the variables.
    """
    return attributes_present(variables, CONVEX_ATTRIBUTES)


def attributes_present(variables, attr_map) -> list[str]:
    """Returns a list of the relevant attributes present
       among the variables.
    """
    return [attr for attr in attr_map if any(v.attributes[attr] for v
                                             in variables)]


def recover_value_for_variable(variable, lowered_value, project: bool = True):
    if variable.attributes['diag']:
        return sp.diags(lowered_value.flatten(order='F'))
    elif attributes_present([variable], SYMMETRIC_ATTRIBUTES):
        n = variable.shape[0]
        value = np.zeros(variable.shape)
        idxs = np.triu_indices(n)
        value[idxs] = lowered_value.flatten(order='F')
        return value + value.T - np.diag(value.diagonal())
    #TODO keep sparse / return coo_tensor
    elif variable.attributes['sparsity']:
        value = np.zeros(variable.shape)
        value[variable.sparse_idx] = lowered_value
        return value
    elif project:
        return variable.project(lowered_value)
    else:
        return lowered_value


def lower_value(variable, value) -> np.ndarray:
    if attributes_present([variable], SYMMETRIC_ATTRIBUTES):
        return value[np.triu_indices(variable.shape[0])]
    elif variable.attributes['diag']:
        return np.diag(value)
    else:
        return value


class CvxAttr2Constr(Reduction):
    """Expand convex variable attributes into constraints."""

    def __init__(self, problem=None, reduce_bounds: bool = False) -> None:
        """If reduce_bounds, reduce lower and upper bounds on variables."""
        self.reduce_bounds = reduce_bounds
        super(CvxAttr2Constr, self).__init__(problem=problem)

    def reduction_attributes(self) -> List[str]:
        """Returns the attributes that will be reduced."""
        if self.reduce_bounds:
            return CONVEX_ATTRIBUTES
        else:
            return [
                attr for attr in CONVEX_ATTRIBUTES if attr not in BOUND_ATTRIBUTES
            ]

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        if not attributes_present(problem.variables(), CONVEX_ATTRIBUTES):
            return problem, ()

        # The attributes to be reduced.
        reduction_attributes = self.reduction_attributes()

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
                for key in reduction_attributes:
                    if new_attr[key]:
                        new_var = True
                        new_attr[key] = None if key == 'bounds' else False

                if attributes_present([var], SYMMETRIC_ATTRIBUTES):
                    n = var.shape[0]
                    shape = (n*(n+1)//2, 1)
                    upper_tri = Variable(shape, var_id=var.id, **new_attr)
                    upper_tri.set_variable_of_provenance(var)
                    id2new_var[var.id] = upper_tri
                    fill_coeff = Constant(upper_tri_to_full(n))
                    full_mat = fill_coeff @ upper_tri
                    obj = reshape(full_mat, (n, n), order='F')
                elif var.attributes['sparsity']:
                    n = len(var.sparse_idx[0])
                    sparse_var = Variable(n, var_id=var.id, **new_attr)
                    sparse_var.set_variable_of_provenance(var)
                    id2new_var[var.id] = sparse_var
                    row_idx = np.ravel_multi_index(var.sparse_idx, var.shape, order='F')
                    col_idx = np.arange(n)
                    coeff_matrix = Constant(sp.csc_matrix((np.ones(n), (row_idx, col_idx)),
                                                    shape=(np.prod(var.shape, dtype=int), n)),
                                                    name="sparse_coeff")
                    obj = reshape(coeff_matrix @ sparse_var, var.shape, order='F')
                elif var.attributes['diag']:
                    diag_var = Variable(var.shape[0], var_id=var.id, **new_attr)
                    diag_var.set_variable_of_provenance(var)
                    id2new_var[var.id] = diag_var
                    obj = diag(diag_var)
                elif new_var:
                    obj = Variable(var.shape, var_id=var.id, **new_attr)
                    obj.set_variable_of_provenance(var)
                    id2new_var[var.id] = obj
                else:
                    obj = var
                    id2new_var[var.id] = obj

                id2new_obj[id(var)] = obj
                # Attributes related to positive and negative definiteness.
                if var.is_psd():
                    constr.append(obj >> 0)
                elif var.attributes['NSD']:
                    constr.append(obj << 0)
                # Add in constraints from bounds.
                if self.reduce_bounds:
                    var._bound_domain(obj, constr)

        # Create new problem.
        obj = problem.objective.tree_copy(id_objects=id2new_obj)
        cons_id_map = {}
        for cons in problem.constraints:
            constr.append(cons.tree_copy(id_objects=id2new_obj))
            cons_id_map[cons.id] = constr[-1].id
        inverse_data = (id2new_var, id2old_var, cons_id_map)
        return cvxtypes.problem()(obj, constr), inverse_data

    def invert(self, solution, inverse_data) -> Solution:
        if not inverse_data:
            return solution

        id2new_var, id2old_var, cons_id_map = inverse_data
        pvars = {}
        for id, var in id2old_var.items():
            new_var = id2new_var[id]
            if new_var.id in solution.primal_vars:
                pvars[id] = recover_value_for_variable(
                    var, solution.primal_vars[new_var.id])

        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in cons_id_map.items()
                 if vid in solution.dual_vars}
        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)
