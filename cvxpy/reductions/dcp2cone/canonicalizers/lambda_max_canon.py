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
    t_expanded = reshape(t, batch_shape + (1, 1), order='C')
    # scipy sparse doesn't support 3D, so use dense eye for batched case
    eye_n = np.eye(n) if batch_shape else sp.eye_array(n)
    t_eye = multiply(t_expanded, eye_n)  # broadcasts to (*batch, n, n)

    constr = [PSD(t_eye - A)]

    if not A.is_symmetric():
        constr.append(swapaxes(A, -2, -1) == A)

    return t, constr
