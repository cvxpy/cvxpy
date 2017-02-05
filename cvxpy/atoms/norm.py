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
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.sigma_max import sigma_max
from cvxpy.atoms.pnorm import pnorm


def norm(x, p=2, axis=None):
    """Wrapper on the different norm atoms.

    Parameters
    ----------
    x : Expression or numeric constant
        The value to take the norm of.
    p : int or str, optional
        The type of norm.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    x = Expression.cast_to_const(x)
    # Norms for scalars same as absolute value.
    if p == 1 or x.is_scalar():
        return pnorm(x, 1, axis)
    elif p == "inf":
        return pnorm(x, 'inf', axis)
    elif p == "nuc":
        return normNuc(x)
    elif p == "fro":
        return pnorm(x, 2, axis)
    elif p == 2:
        if axis is None and x.is_matrix():
            return sigma_max(x)
        else:
            return pnorm(x, 2, axis)
    else:
        return pnorm(x, p, axis)
