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

from cvxpy.atoms.elementwise.power import power


def square(x):
    r"""The elementwise square :math:`x^2`.

    .. math::

        f(x) = x^2

    Convex and nonnegative everywhere.

    Domain: :math:`x \in \mathbb{R}`.

    Parameters
    ----------
    x : Expression
        The expression to square.
    """
    return power(x, 2)
