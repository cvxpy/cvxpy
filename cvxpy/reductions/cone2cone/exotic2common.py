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

import numpy as np
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.solution import Solution
from cvxpy.constraints.power import PowerCone3D, PowerConeND

EXOTIC_CONES = {
    PowerConeND: {PowerCone3D}
}
"""
^ An "exotic" cone is defined as any cone that isn't
supported by ParamConeProg. If ParamConeProg is updated
to support more cones, then it may be necessary to change
this file.
"""


def pow_nd_canon(con, args):
    """
    con : PowerConeND
        I can extract metadata from this.
        For example, con.alpha and con.axis.
    args : tuple of length two
        W,z = args[0], args[1]
    """
    alpha, axis = con.get_data()
    alpha = alpha.value
    if axis != 0:
        raise NotImplementedError()
    W, z = args
    if W.ndim == 1:
        W = reshape(W, (W.size, 1))
        # If this works, then can probably take a
        # transpose to handle the axis argument.
    n, k = W.shape
    if n == 2:
        can_con = PowerCone3D(W[0, :], W[1, :], z, alpha[0, :])
    else:
        T = Variable(shape=(n-2, k))
        arg1, arg2, arg3, arg4 = [], [], [], []
        for j in range(k):
            arg1.append(W[:-1, j])
            arg2.append(T[:, j])
            arg2.append(W[n-1, j])
            arg3.append(z[j])
            arg3.append(T[:, j])
            r_nums = alpha[:, j]
            r_dens = np.cumsum(r_nums)[::-1]  # reverse the cumsum
            r = r_nums / r_dens
            arg4.append(r[:n-1])
        arg1 = hstack(arg1)
        arg2 = hstack(arg2)
        arg3 = hstack(arg3)
        arg4 = hstack(arg4)
        can_con = PowerCone3D(arg1, arg2, arg3, arg4)
    # Return a single PowerCone3D constraint defined over all auxiliary
    # variables needed for the reduction to go through.
    # There are no "auxiliary constraints" beyond this one.
    return can_con, []


class Exotic2Common(Canonicalization):

    CANON_METHODS = {
        PowerConeND: pow_nd_canon
    }

    def __init__(self, problem=None):
        super(Exotic2Common, self).__init__(
            problem=problem, canon_methods=Exotic2Common.CANON_METHODS)

    def accepts(self, problem):
        return True
