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
from cvxpy.utilities.bounds import get_expr_bounds


# If a user insert x * x where x is a variable it gets canonicalized to 
# square(x) before this function is called.
def multiply_canon(expr, args):
    t1 = args[0]
    t2 = args[1]
    constraints = []

    # if either is constant, no canonicalization needed
    if t1.is_constant() or t2.is_constant():
        return expr.copy([t1, t2]), []

    if not isinstance(t1, Variable):
        bounds1 = get_expr_bounds(t1)
        t1 = Variable(t1.shape, bounds=bounds1)
        constraints += [t1 == args[0]]
        t1.value = args[0].value

    if not isinstance(t2, Variable):
        bounds2 = get_expr_bounds(t2)
        t2 = Variable(t2.shape, bounds=bounds2)
        constraints += [t2 == args[1]]
        t2.value = args[1].value

    return expr.copy([t1, t2]), constraints

def matmul_canon(expr, args):
    t1 = args[0]
    t2 = args[1]
    constraints = []

    # if either is constant, no canonicalization needed
    if t1.is_constant() or t2.is_constant():
        return expr.copy([t1, t2]), []

    if not isinstance(t1, Variable):
        bounds1 = get_expr_bounds(t1)
        t1 = Variable(t1.shape, bounds=bounds1)
        constraints += [t1 == args[0]]
        t1.value = args[0].value

    if not isinstance(t2, Variable):
        bounds2 = get_expr_bounds(t2)
        t2 = Variable(t2.shape, bounds=bounds2)
        constraints += [t2 == args[1]]
        t2.value = args[1].value

    return expr.copy([t1, t2]), constraints
