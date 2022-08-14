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

import cvxpy as cp
from cvxpy import Variable
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.zero import Zero


def form_cone_constraint(z: Variable, constraint: Constraint) -> Constraint:
    """
    Given a constraint represented as Ax+b in K for K a cvxpy cone, return an
    instantiated cvxpy constraint.
    """
    if isinstance(constraint, SOC):
        # TODO: Figure out how to instantiate Ax+b in SOC where we know which
        # lines from our ultimate A_pers(x,t,s) + b in K times ... correspond
        # to this constraint.
        return SOC(t=z[0], X=z[1:])
    elif isinstance(constraint, NonNeg):
        return NonNeg(z)
    elif isinstance(constraint, ExpCone):
        n = z.shape[0]
        assert len(z.shape) == 1
        assert n % 3 == 0  # we think this is how the exponential cone works
        step = n//3
        return ExpCone(z[:step], z[step:-step], z[-step:])
    elif isinstance(constraint, Zero):
        return Zero(z)
    elif isinstance(constraint, PSD):
        assert len(z.shape) == 1
        N = z.shape[0]
        n = int(N**.5)
        assert N == n**2, "argument is not a vectorized square matrix"
        z_mat = cp.reshape(z, (n, n))
        return PSD(z_mat)  # do we need constraint_id?
    elif isinstance(constraint, PowCone3D):
        raise NotImplementedError
    else:
        raise NotImplementedError
