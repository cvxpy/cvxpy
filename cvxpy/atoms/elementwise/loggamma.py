"""
Copyright 2021 the CVXPY developers
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

from cvxpy.atoms.elementwise.entr import entr
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.maximum import maximum


# ruff: noqa: E501
def loggamma(x):
    """Elementwise log of the gamma function.

    Implementation has modest accuracy over the full range, approaching perfect
    accuracy as x goes to infinity. For details on the nature of the approximation,
    refer to `CVXPY GitHub Issue #228 <https://github.com/cvxpy/cvxpy/issues/228#issuecomment-544281906>`_.
    """

    return maximum(
        2.18382 - 3.62887*x,
        1.79241 - 2.4902*x,
        1.21628 - 1.37035*x,
        0.261474 - 0.28904*x,
        0.577216 - 0.577216*x,
        -0.175517 + 0.03649*x,
        -1.27572 + 0.621514*x,
        -0.845568 + 0.422784*x,
        -0.577216*x - log(x),
        0.918939 - x - entr(x) - 0.5*log(x),
    )
