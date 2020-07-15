"""
Copyright, the CVXPY authors

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
from cvxpy.atoms.geo_mean import geo_mean
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.inv_pos import inv_pos


def inv_prod(value):
    """The reciprocal of a product of the entries of a vector ``x``.

    Parameters
    ----------
    x : Expression or numeric
        The expression whose reciprocal product is to be computed. Must have
        positive entries.

    Returns
    -------
    Expression
        .. math::
            \\left(\\prod_{i=1}^n x_i\\right)^{-1},

        where :math:`n` is the length of :math:`x`.
    """
    return power(inv_pos(geo_mean(value)), sum(value.shape))
