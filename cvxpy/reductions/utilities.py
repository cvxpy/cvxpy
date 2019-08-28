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

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.zero import Zero
from cvxpy.constraints.nonpos import NonPos
from collections import defaultdict
import numpy as np
import scipy.sparse as sp


def lower_inequality(inequality):
    lhs = inequality.args[0]
    rhs = inequality.args[1]
    return NonPos(lhs - rhs, constr_id=inequality.constr_id)


def lower_equality(equality):
    lhs = equality.args[0]
    rhs = equality.args[1]
    return Zero(lhs - rhs, constr_id=equality.constr_id)


def special_index_canon(expr, args):
    select_mat = expr._select_mat
    final_shape = expr._select_mat.shape
    select_vec = np.reshape(select_mat, select_mat.size, order='F')
    # Select the chosen entries from expr.
    arg = args[0]
    identity = sp.eye(arg.size).tocsc()
    lowered = reshape(identity[select_vec]*vec(arg), final_shape)
    return lowered, []


def are_args_affine(constraints):
    return all(arg.is_affine() for constr in constraints
               for arg in constr.args)


def group_constraints(constraints):
    """Organize the constraints into a dictionary keyed by constraint names.

    Paramters
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


def dict_mat_mul(lh_dm, rh_dm):
    """Multiply two dict mats.

    Parameters
    ----------
    lh_dm : A dictionary of matrices.
    rh_dm : A dictionary of matrices.

    Returns
    -------
    NumPy ndarray
        The product of the dict mats.
    """
    result = 0
    # result = {}
    for lh_key, lh_mat in lh_dm.items():
        for rh_key, rh_mat in rh_dm.items():
            # # Imagine an elementwise matrix product
            # # of two dictionaries.
            # Imagine a matrix dot product
            # of two dictionaries.
            if lh_key == rh_key:
                prod_mat = lh_mat*rh_mat
                # if rh_key not in result:
                #     result[rh_key] = prod_mat
                # else:
                #     result[rh_key] += prod_mat
                result += prod_mat
    return result


def acc_dict_mat(lh_dm, rh_dm):
    """Accumulate right hand dict mat into left hand by addition.

    Parameters
    ----------
    lh_dm : A dictionary of matrices.
    rh_dm : A dictionary of matrices.
    """
    for rh_key, rh_mat in rh_dm.items():
        if rh_key not in lh_dm:
            lh_dm[rh_key] = rh_mat
        else:
            lh_dm[rh_key] += rh_mat


def tensor_mul(lh_ten, rh_dm):
    """Multiply a tensor by a dict mat.

    Parameters
    ----------
    lh_ten : A dict of dict mats.
    rh_dm : A dict mat.

    Returns
    -------
    dict mat
        A dictionary of matrices.
    """
    result = {}
    for lh_key, lh_dm in lh_ten.items():
        # for rh_key, rh_dm in rh_ten.items():
            prod_dm = dict_mat_mul(lh_dm, rh_dm)
            if lh_key not in result:
                result[lh_key] = prod_dm
            else:
                acc_dict_mat(result[lh_key], prod_dm)
    return result
