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

from cvxpy.atoms.affine.bmat import bmat
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable


def sigma_max_canon(expr, args):
    A = args[0]
    n, m = A.shape
    shape = expr.shape
    if not np.prod(shape) == 1:
        raise RuntimeError('Invalid shape of expr in sigma_max canonicalization.')
    t = Variable(shape)
    tI_n = t * sp.eye_array(n)
    tI_m = t * sp.eye_array(m)
    X = bmat([[tI_n, A],
              [A.T, tI_m]])
    constraints = [PSD(X)]
    return t, constraints
