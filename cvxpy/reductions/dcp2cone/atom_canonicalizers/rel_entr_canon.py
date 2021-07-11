"""
Copyright 2021 The CVXPY Developers

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

from cvxpy.atoms.affine.promote import promote
from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.variable import Variable


def rel_entr_canon(expr, args):
    shape = expr.shape
    x = promote(args[0], shape)
    y = promote(args[1], shape)
    t = Variable(shape)
    constraints = [ExpCone(t, x, y)]
    obj = -t
    return obj, constraints
