"""
Copyright 2017 Robin Verschueren

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

from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable


def power_canon(expr, args):
    affine_expr = args[0]
    p = float(expr.p.value)
    if expr.is_constant():
        return Constant(expr.value), []
    elif p == 0:
        return np.ones(affine_expr.shape), []
    elif p == 1:
        return affine_expr, []
    elif p == 2:
        if isinstance(affine_expr, Variable):
            return SymbolicQuadForm(affine_expr, sp.eye(affine_expr.size), expr), []
        else:
            t = Variable(affine_expr.shape)
            return SymbolicQuadForm(t, sp.eye(t.size), expr), [affine_expr == t]
    raise ValueError("non-constant quadratic forms can't be raised to a power "
                     "greater than 2.")
