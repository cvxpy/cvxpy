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

import numpy as np
import cvxpy
from cvxpy.expressions.expression import Expression
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.sigma_max import sigma_max
from cvxpy.atoms.pnorm import pnorm
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.affine.vec import vec


def norm(x, p=2, axis=None):
    """Wrapper on the different norm atoms.

    Parameters
    ----------
    x : Expression or numeric constant
        The value to take the norm of.
    p : int or str, optional
        The type of norm.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    x = Expression.cast_to_const(x)
    # matrix norms take precedence
    if axis is None and x.ndim == 2:
        if p == 1:  # matrix 1-norm
            return cvxpy.atoms.max(norm1(x, axis=0))
        elif p == 2:  # matrix 2-norm is largest singular value
            return sigma_max(x)
        elif p == 'nuc':  # the nuclear norm (sum of singular values)
            return normNuc(x)
        elif p == 'fro':  # Frobenius norm
            return pnorm(vec(x), 2)
        elif p in [np.inf, "inf", "Inf"]:  # the matrix infinity-norm
            return cvxpy.atoms.max(norm1(x, axis=1))
        else:
            raise RuntimeError('Unsupported matrix norm.')
    else:
        if p == 1 or x.is_scalar():
            return norm1(x, axis=axis)
        elif p in [np.inf, "inf", "Inf"]:
            return norm_inf(x, axis)
        else:
            return pnorm(x, p, axis)
