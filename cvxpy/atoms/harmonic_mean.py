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

import numpy as np

from cvxpy.atoms.pnorm import pnorm
from cvxpy.expressions.expression import Expression


def harmonic_mean(x) -> Expression:
    """The harmonic mean of ``x``.

    Parameters
    ----------
    x : Expression or numeric
        The expression whose harmonic mean is to be computed. Must have
        positive entries.

    Returns
    -------
    Expression
        .. math::
            \\frac{n}{\\left(\\sum_{i=1}^{n} x_i^{-1} \\right)},

        where :math:`n` is the length of :math:`x`.

    Raises
    ------
    ValueError
        If ``x`` is known to have negative entries, either because it is
        built from numeric data containing a negative entry, or because
        it is provably nonpositive. Arguments of unknown sign are
        permitted; their nonnegativity is enforced as a domain constraint
        at solve time.
    """
    x = Expression.cast(x)
    # The concave reformulation x.size * pnorm(x, -1) is only valid on the
    # nonnegative orthant. Outside that domain it silently evaluates to
    # -inf instead of the true harmonic mean, so fail fast when the
    # argument is known to have negative entries.
    if x.is_constant() and not x.parameters() and x.value is not None:
        value = np.asarray(x.value)
        has_negative_entry = not np.iscomplexobj(value) and bool(np.any(value < 0))
    else:
        # A nonpositive argument that is not provably zero must contain a
        # negative entry.
        has_negative_entry = x.is_nonpos() and not x.is_nonneg()
    if has_negative_entry:
        raise ValueError(
            "harmonic_mean is only defined for arguments with positive "
            "entries, but the argument is known to have negative entries."
        )
    return x.size*pnorm(x, -1)
