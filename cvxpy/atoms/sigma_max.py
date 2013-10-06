"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from atom import Atom
from .. import utilities as u
from .. import interface as intf
from ..expressions.constants import Constant
from ..expressions.variables import Variable
from ..constraints.affine import AffEqConstraint, AffLeqConstraint
from ..constraints.semi_definite import SDP
from ..interface import numpy_wrapper as np

class sigma_max(Atom):
    """ Maximum singular value. """
    def __init__(self, A):
        super(sigma_max, self).__init__(A)

    # Resolves to a scalar.
    def set_shape(self):
        self._shape = u.Shape(1,1)

    # Always unknown.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.Monotonicity.NONMONOTONIC]
    
    @staticmethod
    def graph_implementation(var_args, size):
        A = var_args[0] # m by n matrix.
        n,m = A.size
        # Create a matrix with Schur complement I*t - (1/t)*A.T*A.
        X = Variable(n+m, n+m)
        t = Variable().canonical_form()[0]
        I_n = Constant(np.eye(n)).canonical_form()[0]
        I_m = Constant(np.eye(m)).canonical_form()[0]
        # Expand A.T.
        obj,constraints = A.T
        # Fix X using the fact that A must be affine by the DCP rules.
        constraints += [AffEqConstraint(X[0:n,0:n], I_n*t),
                        AffEqConstraint(X[0:n,n:n+m], A),
                        AffEqConstraint(X[n:n+m,0:n], obj),
                        AffEqConstraint(X[n:n+m,n:n+m], I_m*t),
        ]
        # Add SDP constraint.
        return (t, [SDP(X)] + constraints)