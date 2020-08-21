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

from cvxpy.expressions.constants import Constant
from cvxpy.constraints.zero import Equality, Zero
import numpy as np


def equality_canon(expr, real_args, imag_args, real2imag):
    if imag_args[0] is None and imag_args[1] is None:
        return [expr.copy(real_args)], None

    # Fill in missing args with zeros.
    for i in range(len(imag_args)):
        if imag_args[i] is None:
            imag_args[i] = Constant(np.zeros(real_args[i].shape))

    imag_cons = [Equality(imag_args[0], imag_args[1],
                          constr_id=real2imag[expr.id])]
    if real_args[0] is None and real_args[1] is None:
        return None, imag_cons
    else:
        # Fill in missing args with zeros.
        for i in range(len(real_args)):
            if real_args[i] is None:
                real_args[i] = Constant(np.zeros(imag_args[i].shape))
        return [expr.copy(real_args)], imag_cons


def zero_canon(expr, real_args, imag_args, real2imag):
    if imag_args[0] is None:
        return [expr.copy(real_args)], None

    imag_cons = [Zero(imag_args[0], constr_id=real2imag[expr.id])]
    if real_args[0] is None:
        return None, imag_cons
    else:
        return [expr.copy(real_args)], imag_cons
