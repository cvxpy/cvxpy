"""
Copyright 2022, the CVXPY authors

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

from cvxpy.atoms.affine.bmat import bmat
from cvxpy.expressions.variable import Variable


def tr_inv_canon(expr, args):
    """Reduces the atom to an affine expression and list of constraints.

    Creates the equivalent problem::

       maximize    sum(u[i])
       subject to: [X ei; ei.T u[i]] is positive semidefinite

       where ei is the n dimensional column vector whose i-th entry is 1 and other entries are 0

    This follows from the inequality:

    .. math::

       u[i] >= R[i][i] for all i, where R=X^-1

    Parameters
    ----------
    expr : tr_inv
    args : list
        The arguments for the expression

    Returns
    -------
    tuple
        (Variable for objective, list of constraints)
    """
    X = args[0]
    n, _ = X.shape
    su = None
    constraints = []
    for i in range(n):
        ei = np.zeros((n, 1))
        ei[i] = 1.0
        ui = Variable((1, 1))
        R = bmat([[X, ei],
                  [ei.T, ui]])
        constraints += [R >> 0]
        if su is None:
            su = ui
        else:
            su += ui
    return su, constraints
