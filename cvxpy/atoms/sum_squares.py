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

from cvxpy.atoms.quad_over_lin import quad_over_lin


def sum_squares(expr):
    """The sum of the squares of the entries.

    Parameters
    ----------
    expr: Expression
        The expression to take the sum of squares of.

    Returns
    -------
    Expression
        An expression representing the sum of squares.
    """
    return quad_over_lin(expr, 1)
