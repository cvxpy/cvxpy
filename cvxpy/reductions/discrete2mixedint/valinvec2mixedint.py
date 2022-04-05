#!/usr/bin/env python
"""
Copyright, the CVXPY authors

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
from cvxpy.constraints import finiteSet
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization


def exprval_in_vec(expr, vec):
    assert expr.is_affine()
    assert expr.size == 1
    if vec.size == 1:
        # handling for when vec only has a single element
        cons = expr == vec[0]
        return [cons]
    vec = np.sort(vec)
    d = np.diff(vec)
    z = Variable(shape=(d.size,), boolean=True)
    cons = [z[i+1] <= z[i] for i in range(d.size-1)]
    cons.append(expr == vec[0] + d @ z)
    return cons


def finite_set_canon(con, args):
    vec = con.vec.value
    expr = args[0]
    cons = exprval_in_vec(expr, vec)
    main_con = cons[-1]
    aux_cons = cons[:-1]
    return main_con, aux_cons


class Valinvec2mixedint(Canonicalization):
    CANON_METHODS = {
        finiteSet: finite_set_canon
    }

    def __init__(self, problem=None) -> None:
        super(Valinvec2mixedint, self).__init__(problem=problem,
                                                canon_methods=Valinvec2mixedint.CANON_METHODS)
