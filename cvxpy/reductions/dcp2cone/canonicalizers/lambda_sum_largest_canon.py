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

from cvxpy.atoms import trace
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.canonicalizers.lambda_max_canon import (
    lambda_max_canon,
)
from cvxpy.utilities.solver_context import SolverInfo


def lambda_sum_largest_canon(expr, args, solver_context: SolverInfo | None = None):
    """
    S_k(X) denotes lambda_sum_largest(X, k)
    t >= k S_k(X - Z) + trace(Z), Z is PSD
    implies
    t >= ks + trace(Z)
    Z is PSD
    sI >= X - Z (PSD sense)
    which implies
    t >= ks + trace(Z) >= S_k(sI + Z) >= S_k(X)
    We use the fact that
    S_k(X) = sup_{sets of k orthonormal vectors u_i}sum_{i}u_i^T X u_i
    and if Z >= X in PSD sense then
    sum_{i}u_i^T Z u_i >= sum_{i}u_i^T X u_i

    We have equality when s = lambda_k and Z diagonal
    with Z_{ii} = (lambda_i - lambda_k)_+
    """
    X = expr.args[0]
    k = expr.k
    n = X.shape[-1]
    if X.ndim == 2:
        Z = Variable((n, n), PSD=True)
        obj, constr = lambda_max_canon(expr, [X - Z])
        obj = k * obj + trace(Z)
        return obj, constr
    else:
        # nd case: X has shape (*batch, n, n)
        batch_shape = X.shape[:-2]
        m = batch_shape[0]  # Only support 1 batch dim for now
        objs = []
        constr = []
        for i in range(m):
            Z_i = Variable((n, n), PSD=True)
            obj_i, constr_i = lambda_max_canon(expr, [X[i] - Z_i])
            objs.append(k * obj_i + trace(Z_i))
            constr.extend(constr_i)
        from cvxpy.atoms.affine.hstack import hstack
        obj = hstack(objs)
        return obj, constr
