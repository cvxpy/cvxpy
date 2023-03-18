"""
Copyright 2022, the CVXPY authors

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
from cvxpy import Variable, lambda_sum_largest, trace
from cvxpy.atoms.affine.sum import sum
from cvxpy.constraints.nonpos import NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.reductions.dcp2cone.atom_canonicalizers.entr_canon import entr_canon
from cvxpy.reductions.dcp2cone.atom_canonicalizers.lambda_sum_largest_canon import (
    lambda_sum_largest_canon,
)


def von_neumann_entr_canon(expr, args):
    N = args[0]
    assert N.is_real()
    n = N.shape[0]
    x = Variable(shape=(n,))
    t = Variable()

    # START code that applies to all spectral functions #
    constrs = []
    for r in range(1, n):
        # lambda_sum_largest(N, r) <= sum(x[:r])
        expr_r = lambda_sum_largest(N, r)
        epi, cons = lambda_sum_largest_canon(expr_r, expr_r.args)
        constrs.extend(cons)
        con = NonPos(epi - sum(x[:r]))
        constrs.append(con)

    # trace(N) \leq sum(x)
    con = trace(N) == sum(x)
    constrs.append(con)

    # trace(N) == sum(x)
    con = Zero(trace(N) - sum(x))
    constrs.append(con)

    # x[:(n-1)] >= x[1:]
    #   x[0] >= x[1],  x[1] >= x[2], ...
    con = NonPos(x[1:] - x[:(n - 1)])
    constrs.append(con)

    # END code that applies to all spectral functions #

    # sum(entr(x)) >= t
    hypos, entr_cons = entr_canon(x, [x])
    constrs.extend(entr_cons)
    con = NonPos(t - sum(hypos))
    constrs.append(con)

    return t, constrs
