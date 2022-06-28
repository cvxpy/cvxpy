"""
Copyright 2021 the CVXPY developers

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

import math
from typing import List, Tuple
import numpy as np

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import sREC
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from scipy.integrate import fixed_quad
import scipy

COMMON_CONES = {
    sREC: {quad_over_lin}
}

def gauss_legendre(n) -> Tuple[np.array, np.array]:
    """
    Helper function for returning the weights and nodes for an
    n-point Gauss-Legendre quadrature
    """
    I = fixed_quad(scipy.log, a=0, b=1, n=n)
    beta = 0.5
    #TODO: Finish this implementation


def sREC_canon(con: sREC, args) -> Tuple[List[Constraint], List]:
    """
    con: sREC
    args:
    """
    vars = dict()
    vars["Z"+str(0)] = con.y
    cons = []

    for i in range(con.k):
        vars["Z"+str(i+1)] = Variable()
        cons.append(vars["Z"+str(i)] >= quad_over_lin(vars["Z"+str(i+1)], con.x))

    W, T = gauss_legendre(con.m)

    for i in range(con.m):
        vars["T"+str(i)] = Variable()
        cons.append(vars["Z"+str(k)]-con.x-vars["T"+str(i)] >=
                    quad_over_lin(-math.sqrt(T[i])*vars["T"+str(i)], con.x-T[i]*vars["T"+str(i)]))

    return cons, []

class Common2Common(Canonicalization):

    CANON_METHODS = {
        sREC: sREC_canon
    }

    def __init__(self, problem=None) -> None:
        super(Common2Common, self).__init__(
            problem=problem, canon_methods=Common2Common.CANON_METHODS)
