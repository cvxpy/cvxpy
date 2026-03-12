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
import scipy.sparse as sp

from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.transpose import swapaxes
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo


def lambda_max_canon(expr, args, solver_context: SolverInfo | None = None):
    A = args[0]
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    t = Variable(batch_shape)

    # Build t * I_n as (*batch, n, n):
    # reshape t to (*batch, 1, 1), elementwise multiply with eye(n)
    t_expanded = reshape(t, batch_shape + (1, 1), order='F')
    # Construct eye(n) as ND COO array with shape (*batch, n, n)
    diag = np.arange(n)
    batch_size = int(np.prod(batch_shape)) if batch_shape else 1
    data = np.ones(batch_size * n)
    rows = np.tile(diag, batch_size)
    cols = np.tile(diag, batch_size)
    if batch_shape:
        batch_flat = np.repeat(np.arange(batch_size), n)
        coords = np.unravel_index(batch_flat, batch_shape) + (rows, cols)
    else:
        coords = (rows, cols)
    eye_n = sp.coo_array((data, coords), shape=batch_shape + (n, n))
    t_eye = multiply(t_expanded, eye_n)  # (*batch, n, n)

    constr = [PSD(t_eye - A)]

    if not A.is_symmetric():
        constr.append(upper_tri(A) == upper_tri(swapaxes(A, -2, -1)))

    return t, constr
