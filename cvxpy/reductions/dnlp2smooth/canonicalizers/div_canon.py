"""
Copyright 2025 CVXPY developers

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

from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.warn import warn

MIN_INIT = 1e-3

# We canonicalize div(f(x), g(x)) as f(x) * power(z, -1), z = g(x), z >= 0.
def div_canon(expr, args):
    if not args[1].is_nonneg():
        warn(
            "CVXPY (DNLP) could not verify that the denominator of a division "
            "appearing in your problem is nonnegative. The solver will proceed "
            "under the assumption that the denominator is nonnegative. If this "
            "assumption is incorrect, the solution may be invalid. "
        )
    
    z = Variable(args[1].shape, nonneg=True)

    if args[1].value is not None:
        z.value = np.maximum(args[1].value, MIN_INIT)

    return multiply(args[0], power(z, -1)), [z == args[1]]
