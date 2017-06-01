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

"""
This script finds a PSD matrix that is closest to a given symmetric,
real matrix, as measured by the Frobenius norm. That is, for
a given matrix P, it solves:
   minimize   || Z - P ||_F
   subject to Z >= 0

Adapted from an example provided in the SeDuMi documentation and CVX examples.
Unlike those examples, the data is real (not complex) and the result is only
required to be PSD (instead of also Toeplitz)
"""
# import cvxpy as cvx
# import numpy as np
# import cvxopt
#
# # create data P
# P = cvxopt.matrix(np.matrix('4 1 3; 1 3.5 0.8; 3 0.8 1'))
# Z = cvx.Variable(3,3)

# objective = cvx.Minimize( sum(cvx.square(P - Z)) )
# constr = [cvx.constraints.semi_definite.SDP(P)]
# prob = cvx.Problem(objective, constr)
# prob.solve()

import cvxpy as cp
import numpy as np
import cvxopt

# create data P
P = cp.Parameter(3,3)
Z = cp.semidefinite(3)

objective = cp.Minimize( cp.lambda_max(P) - cp.lambda_min(P - Z) )
prob = cp.Problem(objective, [Z >= 0])
P.value = cvxopt.matrix(np.matrix('4 1 3; 1 3.5 0.8; 3 0.8 1'))
prob.solve()
print "optimal value =", prob.value


# [ 4,     1+2*j,     3-j       ; ...
#       1-2*j, 3.5,       0.8+2.3*j ; ...
#       3+j,   0.8-2.3*j, 4         ];
#
# % Construct and solve the model
# n = size( P, 1 );
# cvx_begin sdp
#     variable Z(n,n) hermitian toeplitz
#     dual variable Q
#     minimize( norm( Z - P, 'fro' ) )
#     Z >= 0 : Q;
# cvx_end
