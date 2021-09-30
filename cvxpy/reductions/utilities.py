"""
Copyright 2018 Akshay Agrawal

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

from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.nonpos import NonNeg, NonPos
from cvxpy.constraints.zero import Zero


def lower_ineq_to_nonpos(inequality):
    lhs = inequality.args[0]
    rhs = inequality.args[1]
    return NonPos(lhs - rhs, constr_id=inequality.constr_id)


def lower_ineq_to_nonneg(inequality):
    lhs = inequality.args[0]
    rhs = inequality.args[1]
    return NonNeg(rhs - lhs, constr_id=inequality.constr_id)


def lower_equality(equality):
    lhs = equality.args[0]
    rhs = equality.args[1]
    return Zero(lhs - rhs, constr_id=equality.constr_id)


def nonpos2nonneg(nonpos):
    return NonNeg(-nonpos.expr, constr_id=nonpos.constr_id)


def special_index_canon(expr, args):
    select_mat = expr._select_mat
    final_shape = expr._select_mat.shape
    select_vec = np.reshape(select_mat, select_mat.size, order='F')
    # Select the chosen entries from expr.
    arg = args[0]
    identity = sp.eye(arg.size).tocsc()
    lowered = reshape(identity[select_vec] @ vec(arg), final_shape)
    return lowered, []


def are_args_affine(constraints) -> bool:
    return all(arg.is_affine() for constr in constraints
               for arg in constr.args)


def group_constraints(constraints):
    """Organize the constraints into a dictionary keyed by constraint names.

    Parameters
    ---------
    constraints : list of constraints

    Returns
    -------
    dict
        A dict keyed by constraint types where dict[cone_type] maps to a list
        of exactly those constraints that are of type cone_type.
    """
    constr_map = defaultdict(list)
    for c in constraints:
        constr_map[type(c)].append(c)
    return constr_map
