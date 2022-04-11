#!/usr/bin/env python
"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Callable, List, Tuple

import numpy as np

import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization


def exprval_in_vec_ineq(expr, vec):

    vec = np.sort(vec)
    d = np.diff(vec)
    z = Variable(shape=(d.size,), boolean=True)
    cons = [
        expr == vec[0] + d @ z,
    ]
    if d.size > 1:
        cons.append(z[1:] <= z[:-1])
    return cons


def exprval_in_vec_eq(expr, vec):
    # Reference: https://docs.mosek.com/modeling-cookbook/mio.html#fixed-set-of-values

    z = Variable(len(vec), boolean=True)
    constraints = [
        cp.sum(cp.multiply(vec, z)) == expr,
        cp.sum(z) == 1
    ]
    return constraints


def get_exprval_in_vec_func(ineq_form: bool) -> Callable:
    if ineq_form:
        return exprval_in_vec_ineq
    else:
        return exprval_in_vec_eq


def finite_set_canon(con, _args) -> Tuple[Constraint, List]:
    cons = []
    vec = con.vec.value
    if vec.size == 1:
        # handling for when vec only has a single element
        return con.expre == vec[0], []

    expre = con.expre.flatten()
    for i in range(expre.size):
        exprval_in_vec = get_exprval_in_vec_func(con.ineq_form)
        cons += exprval_in_vec(expre[i], vec)
    main_con = cons[0]
    aux_cons = cons[1:]
    return main_con, aux_cons


class Valinvec2mixedint(Canonicalization):
    def accepts(self, problem) -> bool:
        return any(FiniteSet in {type(c) for c in problem.constraints})

    CANON_METHODS = {
        FiniteSet: finite_set_canon
    }

    def __init__(self, problem=None) -> None:
        super(Valinvec2mixedint, self).__init__(problem=problem,
                                                canon_methods=Valinvec2mixedint.CANON_METHODS)
