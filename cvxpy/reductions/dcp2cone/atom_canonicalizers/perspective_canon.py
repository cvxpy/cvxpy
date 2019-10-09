"""
Copyright 2019 Shane Barratt

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
from cvxpy.reductions.dcp2cone import atom_canonicalizers
import cvxpy as cp
import numpy as np
import IPython as ipy

def perspective_canon(expr, args):
    # perspective(f)(x, t) = {
    #    tf(x/t)  if t > 0,
    #    0        if t = 0, x = 0
    #    infinity otherwise
    # }     

    # f(x) <= s <==> Ax + bs + c \in \mathcal K
    # tf(x/t) <= s <==> Ax + bs + ct \in \mathcal K

    x = args[:-1]
    t = args[-1].flatten()

    underlying_canonicalizer = atom_canonicalizers.CANON_METHODS[type(expr._atom_initialized)]
    s, constraints_underlying = underlying_canonicalizer(expr._atom_initialized, expr._atom_initialized.args)
    s.value = np.zeros(s.shape)

    constraints = []
    for constraint in constraints_underlying:
        constraint_arguments = []

        for arg in constraint.args:
            # set all variables to zero, save values
            var_values = []
            for var in arg.variables():
                if var.is_constant():
                    continue
                var_values.append(var.value[:])
                var.value = np.zeros(var.shape)
            
            # create new constraint for perspective
            constraint_arguments.append(arg + arg.value[:] * (t - 1.0))

            # reset variables to previous values
            for var, value in zip(arg.variables(), var_values):
                if var.is_constant():
                    continue
                var.value = value
        
        constraints += [type(constraint)(*constraint_arguments)]

    return s, constraints
