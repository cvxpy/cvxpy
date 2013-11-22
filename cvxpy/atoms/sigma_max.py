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
from affine.transpose import transpose
from .. import utilities as u
from .. import interface as intf
from ..expressions.constants import Constant
from ..expressions.variables import Variable
from ..constraints.semi_definite import SDP
import numpy as np
from numpy import linalg as LA

class sigma_max(Atom):
    """ Maximum singular value. """
    def __init__(self, A):
        super(sigma_max, self).__init__(A)

    # Returns the largest singular value of A.
    @Atom.numpy_numeric
    def numeric(self, values):
        return LA.norm(values[0], 2)

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1,1)

    # Always unknown.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.monotonicity.NONMONOTONIC]

    def graph_implementation(self, arg_objs):
        A = arg_objs[0] # m by n matrix.
        n,m = A.size
        # Create a matrix with Schur complement I*t - (1/t)*A.T*A.
        X = Variable(n+m, n+m)
        t = Variable()
        I_n = Constant(np.eye(n))
        I_m = Constant(np.eye(m))
        # Expand A.T.
        obj,constr = A.T.canonical_form
        # Fix X using the fact that A must be affine by the DCP rules.
        constr += [X[0:n,0:n] == I_n*t,
                   X[0:n,n:n+m] == A,
                   X[n:n+m,0:n] == obj,
                   X[n:n+m,n:n+m] == I_m*t,
        ]
        # Add SDP constraint.
        return (t, [SDP(X)] + constr)