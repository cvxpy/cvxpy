"""
Copyright 2013 Steven Diamond

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

from typing import Union

import numpy as np

import cvxpy
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.pnorm import pnorm
from cvxpy.atoms.sigma_max import sigma_max
from cvxpy.expressions.expression import Expression


def norm(x, p: Union[int, str] = 2, axis=None):
    """Wrapper on the different norm atoms.

    Parameters
    ----------
    x : Expression or numeric constant
        The value to take the norm of.  If `x` is 2D and `axis` is None,
        this function constructs a matrix norm.
    p : int or str, optional
        The type of norm. Valid options include any positive integer,
        'fro' (for frobenius), 'nuc' (sum of singular values), np.inf or
        'inf' (infinity norm).
    axis : The axis along which to apply the norm, if any.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    x = Expression.cast_to_const(x)
    # matrix norms take precedence
    num_nontrivial_idxs = sum([d > 1 for d in x.shape])
    if axis is None and x.ndim == 2:
        if p == 1:  # matrix 1-norm
            return cvxpy.atoms.max(norm1(x, axis=0))
        # Frobenius norm
        elif p == 'fro' or (p == 2 and num_nontrivial_idxs == 1):
            return pnorm(vec(x), 2)
        elif p == 2:  # matrix 2-norm is largest singular value
            return sigma_max(x)
        elif p == 'nuc':  # the nuclear norm (sum of singular values)
            return normNuc(x)
        elif p in [np.inf, "inf", "Inf"]:  # the matrix infinity-norm
            return cvxpy.atoms.max(norm1(x, axis=1))
        else:
            raise RuntimeError('Unsupported matrix norm.')
    else:
        if p == 1 or x.is_scalar():
            return norm1(x, axis=axis)
        elif str(p).lower() == "inf":
            return norm_inf(x, axis)
        elif str(p).lower() == "fro":
            return pnorm(vec(x), 2, axis)
        elif isinstance(p, str):
            raise RuntimeError(f'Unsupported norm option {p} for non-matrix.')
        else:
            return pnorm(x, p, axis)


def norm2(x, axis=None):
    """The 2-norm of x.

    Parameters
    ----------
    x : Expression or numeric constant
        The value to take the norm of.  If `x` is 2D and `axis` is None,
        this function constructs a matrix norm.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    return norm(x, p=2, axis=axis)
