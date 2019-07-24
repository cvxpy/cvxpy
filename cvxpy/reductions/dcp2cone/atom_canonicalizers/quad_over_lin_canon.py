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

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable


def quad_over_lin_canon(expr, args):
    x = args[0]
    y = args[1]

    if y.ndim == 0:
        # quad_over_lin := sum_{ij} X^2_{ij} / y
        y = y.flatten()
        # precondition: shape == ()
        t = Variable(1,)
        # (y+t, y-t, 2*x) must lie in the second-order cone,
        # where y+t is the scalar part of the second-order
        # cone constraint.
        constraints = [SOC(
                           t=y+t,
                           X=hstack([y-t, 2*x.flatten()]), axis=0
                          )]
        return t, constraints

    if (y.ndim == 1):
        if x.ndim == 1:
            x = reshape(x, (x.size, 1))
        assert x.ndim == 2, "Wrong sizes"
        # quad_over_lin := sum_{i} X^2_{i} / y_{i}
        # precondition: shape == ()
        t = Variable(y.size,)
        assert x.shape[0] == y.size, "Wrong sizes"
        y_minus_t = reshape(y-t, (y.size, 1))
        constraints = [SOC(
                           t=y+t,
                           X=hstack([y_minus_t, 2*x]), axis=1
                          )]
        return t, constraints
