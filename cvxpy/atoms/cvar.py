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
    r"""The Conditional Value at Risk (CVaR) of a discrete random variable.

    CVaR at level beta is a risk measure that captures the expected value over 
    the worst (1-beta) fraction of outcomes of a real-valued random variable. 
    For a random variable X representing losses, CVaR at level beta is defined as:

    .. math::
        \phi_\beta(X) = \mathbb{E}[X | X \geq \psi_\beta(X)]

    where psi_beta(X) is the Value at Risk (VaR) at level beta.

    For a discrete distribution represented by samples z_1, ..., z_m, 
    CVaR can be computed as:

    .. math::
        \phi_\beta(z) = \inf_{\alpha \in \mathbb{R}} \left\{ \alpha + 
        \frac{1}{(1-\beta)m}\sum_{i=1}^m(z_i-\alpha)_+ \right\}

    where (z-alpha)_+ = max(z-alpha, 0) is the positive part of z-alpha.

    This function is a wrapper around the dotsort atom, providing an efficient 
    computation of CVaR for discrete distributions.

    Parameters
    ----------
    x : cvxpy.Expression
        The vector of sample losses.
    beta : float
        The probability level, between 0 and 1.
    """
    k = (1 - beta) * x.shape[0]
    w = np.append(np.ones(int(k)), k - int(k))
    return 1/k * dotsort(x, w)
    
