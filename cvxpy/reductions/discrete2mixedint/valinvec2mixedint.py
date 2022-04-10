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

import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization


def exprval_in_vec1(expr, vec):
    assert expr.is_affine()
    assert expr.size == 1
    if vec.size == 1:
        # handling for when vec only has a single element
        cons = [expr == vec[0]]
        return cons
    vec = np.sort(vec)
    d = np.diff(vec)
    z = Variable(shape=(d.size,), boolean=True)
    cons = [
        z[1:] <= z[:-1],
        expr == vec[0] + d @ z
    ]
    return cons


def exprval_in_vec2(expr, vec):
    z = Variable(len(vec), boolean=True)
    constraints = [
        cp.sum(cp.multiply(vec, z)) == expr,
        cp.sum(z) == 1
    ]
    return constraints


def finite_set_canon1(con, args):
    cons = []
    vec = con.vec.value
    expre = con.expre.flatten()
    for i in range(expre.size):
        cons += exprval_in_vec1(expre[i], vec)
    main_con = cons[-1]
    aux_cons = cons[:-1]
    return main_con, aux_cons


def finite_set_canon2(con, args):
    cons = []
    vec = con.vec.value
    expre = con.expre.flatten()
    for i in range(expre.size):
        cons += exprval_in_vec2(expre[i], vec)
    main_con = cons[0]
    aux_cons = cons[1:]
    return main_con, aux_cons


def finite_set_canon(con, args):
    if con.ineq_form:
        return finite_set_canon1(con, args)
    else:
        return finite_set_canon2(con, args)


class Valinvec2mixedint(Canonicalization):
    CANON_METHODS = {
        FiniteSet: finite_set_canon
    }

    def __init__(self, problem=None) -> None:
        super(Valinvec2mixedint, self).__init__(problem=problem,
                                                canon_methods=Valinvec2mixedint.CANON_METHODS)
