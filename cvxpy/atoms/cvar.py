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

import numpy as np

from cvxpy.atoms.dotsort import dotsort


def cvar(x, beta):
    r"""Conditional value at risk (CVaR) at probability level :math:`\beta` of a vector :math:`x`.
    
    It represents the average of the :math:`(1-\beta)` fraction of largest values in :math:`x`. 
    If a probability distribution is represented by a finite set of samples 
    :math:`x_1, \ldots, x_m \in \mathbb{R}`, the CVaR at level :math:`\beta`, denoted as 
    :math:`\phi_\beta(x): \mathbb{R}^m \rightarrow \mathbb{R}`, can be computed as: 

    .. math::
        \phi_\beta(x) = \inf_{\alpha \in \mathbb{R}} \left\{ \alpha + 
        \frac{1}{(1-\beta)m}\sum_{i=1}^m(x_i-\alpha)_+ \right\}

    where :math:`(x-\alpha)_+ = \max(x-\alpha, 0)` is the positive part of :math:`x-\alpha`.


    Parameters
    ----------
    x : Expression or numeric constant.
        The vector of samples.
    beta : float
        The probability level, must be in the range :math:`[0, 1)`.

    Returns
    -------
    Expression
        The CVaR of :math:`x` at probability level :math:`\beta`.
    """
    if not 0 <= beta < 1:
        raise ValueError(f"The probability level beta must be in the range [0, 1), got {beta}")

    if len(x.shape) != 1:
        raise ValueError(f"cvar input must be a 1d array, got shape {x.shape}")
    
    k = (1 - beta) * x.shape[0]
    w = np.append(np.ones(int(k)), k - int(k))
    return 1/k * dotsort(x, w)
    
