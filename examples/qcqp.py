"""
Copyright 2013 Steven Diamond

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

# for decimal division
from __future__ import division
import sys

import cvxopt
import numpy as np
from pylab import *
import math

from cvxpy import *

# Taken from CVX website http://cvxr.com/cvx/examples/
# Derived from Example: Finding the fastest mixing Markov chain on a graph
# Ported from cvx matlab to cvxpy by Misrab Faizullah-Khan
# Original comments below

# Boyd & Vandenberghe, "Convex Optimization"
# Joelle Skaf - 08/23/05
#
# Solved a QCQP with 3 inequalities:
#           minimize    1/2 x'*P0*x + q0'*r + r0
#               s.t.    1/2 x'*Pi*x + qi'*r + ri <= 0   for i=1,2,3
# and verifies that strong duality holds.

# Input data
n = 6
eps = sys.float_info.epsilon

P0 = cvxopt.normal(n, n)
eye = cvxopt.spmatrix(1.0, range(n), range(n))
P0 = P0.T * P0 + eps * eye

print(P0)

P1 = cvxopt.normal(n, n)
P1 = P1.T*P1
P2 = cvxopt.normal(n, n)
P2 = P2.T*P2
P3 = cvxopt.normal(n, n)
P3 = P3.T*P3

q0 = cvxopt.normal(n, 1)
q1 = cvxopt.normal(n, 1)
q2 = cvxopt.normal(n, 1)
q3 = cvxopt.normal(n, 1)

r0 = cvxopt.normal(1, 1)
r1 = cvxopt.normal(1, 1)
r2 = cvxopt.normal(1, 1)
r3 = cvxopt.normal(1, 1)

# Form the problem
x = Variable(n)
objective = Minimize( 0.5*quad_form(x,P0) + q0.T*x + r0 )
constraints = [ 0.5*quad_form(x,P1) + q1.T*x + r1 <= 0,
                0.5*quad_form(x,P2) + q2.T*x + r2 <= 0,
                0.5*quad_form(x,P3) + q3.T*x + r3 <= 0
               ]

# We now find the primal result and compare it to the dual result
# to check if strong duality holds i.e. the duality gap is effectively zero
p = Problem(objective, constraints)
primal_result = p.solve()

if p.status is OPTIMAL:
    # Note that since our data is random, we may need to run this program multiple times to get a feasible primal
    # When feasible, we can print out the following values
    print (x.value) # solution
    lam1 = constraints[0].dual_value
    lam2 = constraints[1].dual_value
    lam3 = constraints[2].dual_value


    P_lam = P0 + lam1*P1 + lam2*P2 + lam3*P3
    q_lam = q0 + lam1*q1 + lam2*q2 + lam3*q3
    r_lam = r0 + lam1*r1 + lam2*r2 + lam3*r3
    dual_result = -0.5*q_lam.T*P_lam*q_lam + r_lam
    # ISSUE: dual result is matrix for some reason

    print ('Our duality gap is:')
    print (primal_result - dual_result)
