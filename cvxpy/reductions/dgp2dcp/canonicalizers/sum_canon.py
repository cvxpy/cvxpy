"""
Copyright 2024 the CVXPY developers

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
from cvxpy.reductions.dgp2dcp.canonicalizers.add_canon import add_canon
from cvxpy.reductions.dgp2dcp.util import explicit_sum


def sum_canon(expr, args):
    X = args[0]
    if expr.axis is None:
        summation = explicit_sum(X)
        canon, _ = add_canon(summation, summation.args)
        return reshape(canon, expr.shape, order='F'), []

    if expr.axis == 0:
        X = X.T

    rows = []
    for i in range(X.shape[0]):
        summation = explicit_sum(X[i])
        canon, _ = add_canon(summation, summation.args)
        rows.append(canon)
    canon = hstack(rows)
    return reshape(canon, expr.shape, order='F'), []
