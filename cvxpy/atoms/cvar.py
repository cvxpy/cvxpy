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

from cvxpy.atoms.axis_atom import axis_size
from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.expressions.expression import Expression


def cvar(x, beta, axis=None, keepdims=False) -> Expression:
    r"""The conditional value at risk (CVaR) of a random variable represented by
    the vector of samples ``x``.

    For a probability level :math:`\beta \in [0,1)`, CVaR is the expected value of
    ``x`` in the worst :math:`(1-\beta)` fraction of cases. Equivalently, it is the
    average of the :math:`(1-\beta)` fraction of largest values in ``x``.

    Parameters
    ----------
    x : Expression or numeric constant
        The samples representing the distribution. With ``axis`` given, the
        samples of each distribution run along the reduced axes, so an array of
        shape ``(m, n)`` with ``axis=0`` holds ``n`` distributions of ``m``
        samples each.
    beta : float
        The probability level. Must be in the range :math:`[0, 1)`.
        For example, :math:`\beta = 0.95` gives the average of the worst 5% of outcomes.
    axis : int or tuple of ints, optional
        The axis or axes along which the samples are taken. The default,
        ``None``, treats all the entries of ``x`` as one distribution.
    keepdims : bool, optional
        If True, the reduced axes are kept with size one.

    Returns
    -------
    Expression
        .. math::

            \frac{1}{(1-\beta)m} \sum\nolimits_{\text{largest } (1-\beta)m} x_i

        where :math:`m` is the number of samples reduced, that is the size of
        :math:`x` along ``axis``. When :math:`(1-\beta)m` is not an integer, the
        fractional part is handled via linear interpolation.
    """
    if not 0 <= beta < 1:
        raise ValueError(f"The probability level beta must be in the range [0, 1), got {beta}")

    k = (1 - beta) * axis_size(x, axis)
    return sum_largest(x, k, axis=axis, keepdims=keepdims) / k
