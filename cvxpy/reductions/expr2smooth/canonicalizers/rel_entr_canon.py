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

from cvxpy.expressions.variable import Variable
import numpy as np

def rel_entr_canon(expr, args):
    constraints = []

    if not args[0].is_constant():
        t1 = Variable(args[0].shape, bounds=[0, None])
        constraints.append(t1 == args[0])

        if args[0].value is not None and np.all(args[0].value >= 1):
            t1.value = args[0].value
        else:
            t1.value = expr.point_in_domain(argument=0)
    else:
        t1 = args[0]

    if not args[1].is_constant():
        t2 = Variable(args[1].shape, bounds=[0, None])
        constraints.append(t2 == args[1])

        if args[1].value is not None and np.all(args[1].value >= 1):
            t2.value = args[1].value
        else:
            t2.value = expr.point_in_domain(argument=1)
    else:
        t2 = args[1]

    return expr.copy([t1, t2]), constraints
