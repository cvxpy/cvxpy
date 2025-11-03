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

from cvxpy.atoms.sum_largest import sum_largest


def cvar(x, beta):
    r"""The conditional value at risk (CVaR) of a random variable represented by
    the vector of samples ``x``.

    For a probability level :math:`\beta \in [0,1)`, CVaR is the expected value of
    ``x`` in the worst :math:`(1-\beta)` fraction of cases. Equivalently, it is the
    average of the :math:`(1-\beta)` fraction of largest values in ``x``.

    Parameters
    ----------
    x : Expression or numeric constant
        A vector of samples representing the distribution. Must be one-dimensional.
    beta : float
        The probability level. Must be in the range :math:`[0, 1)`.
        For example, :math:`\beta = 0.95` gives the average of the worst 5% of outcomes.

    Returns
    -------
    Expression
        .. math::

            \frac{1}{(1-\beta)m} \sum\nolimits_{\text{largest } (1-\beta)m} x_i

        where :math:`m` is the length of :math:`x`. When :math:`(1-\beta)m` is not an
        integer, the fractional part is handled via linear interpolation.
    """
    if not 0 <= beta < 1:
        raise ValueError(f"The probability level beta must be in the range [0, 1), got {beta}")

    if len(x.shape) != 1:
        raise ValueError(f"cvar input must be a 1d array, got shape {x.shape}")

    k = (1 - beta) * x.shape[0]
    return sum_largest(x, k) / k
    
