"""
Copyright 2017 Steven Diamond

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

from cvxpy.expressions.expression import Expression
from cvxpy.atoms.norm import norm
from cvxpy.atoms.affine.hstack import hstack


def mixed_norm(X, p=2, q=1):
    """Lp,q norm; :math:` (\sum_k (\sum_l \lvert x_{k,l} \rvert )^q/p)^{1/q}`.

    Parameters
    ----------
    X : Expression or numeric constant
        The matrix to take the l_{p,q} norm of.
    p : int or str, optional
        The type of inner norm.
    q : int or str, optional
        The type of outer norm.

    Returns
    -------
    Expression
        An Expression representing the mixed norm.
    """
    X = Expression.cast_to_const(X)

    # inner norms
    vecnorms = [norm(X[i, :], p) for i in range(X.size[0])]

    # outer norm
    return norm(hstack(*vecnorms), q)
