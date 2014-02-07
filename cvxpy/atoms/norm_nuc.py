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

class normNuc(Atom):
    """ Sum of the singular values. """
    def __init__(self, A):
        super(normNuc, self).__init__(A)

    # Returns the nuclear norm (i.e. the sum of the singular values) of A.
    @Atom.numpy_numeric
    def numeric(self, values):
        U,s,V = LA.svd(values[0])
        return sum(s)

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1,1)

    # Always positive.
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
        # Create the equivalent problem:
        #   minimize (trace(U) + trace(V))/2
        #   subject to:
        #            [U A; A.T V] is positive semidefinite
        X = Variable(n+m, n+m)
        # Expand A.T.
        obj, constr = A.T.canonical_form
        # Fix X using the fact that A must be affine by the DCP rules.
        constr += [X[0:n,n:n+m] == A, X[n:n+m,0:n] == obj]
        trace = 0.5*sum([X[i,i] for i in range(n+m)])
        # Add SDP constraint.
        return (trace, [SDP(X)] + constr)
