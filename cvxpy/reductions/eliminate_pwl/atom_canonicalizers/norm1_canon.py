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

from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.abs_canon import (
    abs_canon,)


def norm1_canon(expr, args):
    x = args[0]
    axis = expr.axis

    # we need an absolute value constraint for the symmetric convex branches
    # (p >= 1)
    constraints = []
    # TODO(akshayka): Express this more naturally (recursively), in terms
    # of the other atoms
    abs_expr = abs(x)
    abs_x, abs_constraints = abs_canon(abs_expr, abs_expr.args)
    constraints += abs_constraints
    return sum(abs_x, axis=axis), constraints
