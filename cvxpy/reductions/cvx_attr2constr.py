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
from cvxpy.expressions.constants.parameter import Parameter
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


def recover_value_for_leaf(variable, lowered_value, project: bool = True):
    if variable.attributes['diag']:
        return sp.diags_array(lowered_value.flatten(order='F'))
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


def lower_value(variable, value=None) -> np.ndarray:
    """Extract the reduced representation of a leaf's value.

    Args:
        variable: The leaf whose attributes determine the reduction.
        value: If provided, a full-size value (e.g. a differentiation delta)
            to reduce.  If ``None``, reads the leaf's stored ``_value``.

    Notes:
        Called without *value* by ``update_parameters`` and ``apply`` to read
        the current parameter value into the reduced parameter.  Called *with*
        an explicit value by ``param_forward`` to reduce a full-size delta.

        For sparse leaves ``Leaf.save_value`` already stores only the nonzero
        entries, so when ``value is None`` the sparse branch can return
        ``_value`` directly.  An explicit *value* is always full-size and must
        be extracted at the sparse indices.
    """
    # Track whether the caller supplied a full-size value.  When value is None
    # we read _value, which for sparse leaves is already in reduced form.
    full_size = value is not None
    if value is None:
        value = variable._value
    if attributes_present([variable], SYMMETRIC_ATTRIBUTES):
        return value[np.triu_indices(variable.shape[0])]
    elif variable.attributes['diag']:
        return np.diag(value)
    elif variable.attributes['sparsity']:
        if full_size:
            return np.asarray(value)[variable.sparse_idx]
        else:
            # _value already stores only the nonzero data (see Leaf.save_value).
            return np.asarray(value)
    else:
        return value


def build_dim_reduced_expression(leaf, reduced_leaf):
    """Build Expression that reconstructs full shape from a reduced-size leaf."""
    if attributes_present([leaf], SYMMETRIC_ATTRIBUTES):
        n = leaf.shape[0]
        return reshape(Constant(upper_tri_to_full(n)) @ reduced_leaf, (n, n), order='F')
    elif leaf.sparse_idx is not None:
        n = len(leaf.sparse_idx[0])
        row_idx = np.ravel_multi_index(leaf.sparse_idx, leaf.shape, order='F')
        coeff = Constant(sp.csc_array((np.ones(n), (row_idx, np.arange(n))),
                         shape=(np.prod(leaf.shape, dtype=int), n)), name="sparse_coeff")
        return reshape(coeff @ reduced_leaf, leaf.shape, order='F')
    elif leaf.attributes['diag']:
        return diag(reduced_leaf)
    return reduced_leaf


class CvxAttr2Constr(Reduction):
    """Expand convex variable attributes into constraints."""

    def __init__(self, problem=None, reduce_bounds: bool = False) -> None:
        """If reduce_bounds, reduce lower and upper bounds on variables."""
        self.reduce_bounds = reduce_bounds
        self._parameters = {}  # {orig_param: reduced_param}
        self._variables = {}   # {orig_var: new_var} — only for changed vars
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
        has_var_attrs = attributes_present(problem.variables(), CONVEX_ATTRIBUTES)
        has_param_attrs = any(p._has_dim_reducing_attr for p in problem.parameters())
        if not has_var_attrs and not has_param_attrs:
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

                if var._has_dim_reducing_attr:
                    n = var._reduced_size

                    # Transform bounds for sparse reduced variable
                    if var.attributes['sparsity'] and \
                            'bounds' not in reduction_attributes and new_attr.get('bounds'):
                        bounds = new_attr['bounds']
                        transformed_bounds = []
                        for bound in bounds:
                            if sp.issparse(bound):
                                coo = sp.coo_array(bound)
                                coo.sum_duplicates()
                                transformed_bounds.append(coo.data)
                            elif np.isscalar(bound) or (
                                    hasattr(bound, 'ndim') and bound.ndim == 0):
                                transformed_bounds.append(bound)
                            else:
                                raise ValueError(
                                    "Unexpected dense array bound on sparse "
                                    "variable during reduction."
                                )
                        new_attr['bounds'] = transformed_bounds

                    reduced_var = Variable(n, name=var.name(), **new_attr)
                    self._variables[var] = reduced_var
                    id2new_var[var.id] = reduced_var
                    obj = build_dim_reduced_expression(var, reduced_var)
                elif new_var:
                    obj = Variable(var.shape, name=var.name(), **new_attr)
                    self._variables[var] = obj
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

        # For each unique parameter with dim-reducing attributes, create a
        # reduced parameter and a reconstruction expression.
        for param in problem.parameters():
            if param._has_dim_reducing_attr and id(param) not in id2new_obj:
                n = param._reduced_size
                new_attr = param.attributes.copy()
                for key in reduction_attributes:
                    if new_attr[key]:
                        new_attr[key] = None if key == 'bounds' else False
                reduced_param = Parameter(n, name=param.name(), **new_attr)
                self._parameters[param] = reduced_param
                if param.value is not None:
                    reduced_param.value = lower_value(param)
                obj = build_dim_reduced_expression(param, reduced_param)
                id2new_obj[id(param)] = obj

        # Create new problem.
        obj = problem.objective.tree_copy(id_objects=id2new_obj)
        cons_id_map = {}
        for cons in problem.constraints:
            constr.append(cons.tree_copy(id_objects=id2new_obj))
            cons_id_map[cons.id] = constr[-1].id
        inverse_data = (id2new_var, id2old_var, cons_id_map)
        return cvxtypes.problem()(obj, constr), inverse_data

    def update_parameters(self, problem) -> None:
        """Update reduced parameter values from original parameters."""
        for param, reduced_param in self._parameters.items():
            if param.value is not None:
                reduced_param.value = lower_value(param)

    def var_forward(self, dvars):
        """Transform variable deltas from inner (reduced) to outer (original)."""
        result = dict(dvars)
        for orig_var, new_var in self._variables.items():
            if new_var.id in result:
                value = result.pop(new_var.id)
                if orig_var._has_dim_reducing_attr:
                    value = recover_value_for_variable(orig_var, value, project=False)
                result[orig_var.id] = value
        return result

    def var_backward(self, del_vars):
        """Transform variable gradients from outer (original) to inner (reduced)."""
        result = dict(del_vars)
        for orig_var, new_var in self._variables.items():
            if orig_var.id in result:
                value = result.pop(orig_var.id)
                if orig_var._has_dim_reducing_attr:
                    if attributes_present([orig_var], SYMMETRIC_ATTRIBUTES):
                        value = value + value.T - np.diag(np.diag(value))
                    value = lower_value(orig_var, value)
                result[new_var.id] = value
        return result

    def param_backward(self, dparams):
        """Recover full-size gradients from reduced-size gradients."""
        result = dict(dparams)
        for param, reduced_param in self._parameters.items():
            if reduced_param.id in result:
                result[param.id] = recover_value_for_variable(
                    param, result.pop(reduced_param.id))
        return result

    def param_forward(self, param_deltas):
        """Transform full-size deltas to reduced-size deltas."""
        result = dict(param_deltas)
        for param, reduced_param in self._parameters.items():
            if param.id in result:
                result[reduced_param.id] = lower_value(param, result.pop(param.id))
        return result

    def invert(self, solution, inverse_data) -> Solution:
        if not inverse_data:
            return solution

        id2new_var, id2old_var, cons_id_map = inverse_data
        pvars = {}
        for id, var in id2old_var.items():
            new_var = id2new_var[id]
            if new_var.id in solution.primal_vars:
                pvars[id] = recover_value_for_leaf(
                    var, solution.primal_vars[new_var.id])

        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in cons_id_map.items()
                 if vid in solution.dual_vars}
        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)
