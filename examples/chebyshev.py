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

from __future__ import division

import cvxopt
import numpy as np
from pylab import *
import math

# from cvxpy import numpy as my_numpy

from cvxpy import *

# Taken from CVX website http://cvxr.com/cvx/examples/
# Example: Compute and display the Chebyshev center of a 2D polyhedron
# Ported from cvx matlab to cvxpy by Misrab Faizullah-Khan
# Original comments below

# Boyd & Vandenberghe, "Convex Optimization"
# Joelle Skaf - 08/16/05
# (a figure is generated)
#
# The goal is to find the largest Euclidean ball (i.e. its center and
# radius) that lies in a polyhedron described by linear inequalities in this
# fashion: P = { x : a_i'*x <= b_i, i=1,...,m } where x is in R^2

# Create the problem

# variables
radius = Variable(1)
center = Variable(2)

# constraints
a1 = cvxopt.matrix([2,1], (2,1))
a2 = cvxopt.matrix([2,-1], (2,1))
a3 = cvxopt.matrix([-1,2], (2,1))
a4 = cvxopt.matrix([-1,-2], (2,1))

b = cvxopt.matrix(1, (4,1))


constraints = [ a1.T*center + np.linalg.norm(a1, 2)*radius <= b[0],
				a2.T*center + np.linalg.norm(a2, 2)*radius <= b[1],
				a3.T*center + np.linalg.norm(a3, 2)*radius <= b[2],
				a4.T*center + np.linalg.norm(a4, 2)*radius <= b[3] ]


# objective
objective = Maximize(radius)

p = Problem(objective, constraints)
# The optimal objective is returned by p.solve().
result = p.solve()
# The optimal value
print(radius.value)
print(center.value)
# Convert to 1D array.
center_val = np.asarray(center.value[:,0])


# Now let's plot it
x = np.linspace(-2, 2, 256,endpoint=True)
theta = np.linspace(0,2*np.pi,100)

# plot the constraints
plot( x, -x*a1[0]/a1[1] + b[0]/a1[1])
plot( x, -x*a2[0]/a2[1] + b[0]/a2[1])
plot( x, -x*a3[0]/a3[1] + b[0]/a3[1])
plot( x, -x*a4[0]/a4[1] + b[0]/a4[1])


# plot the solution
plot( center_val[0] + radius.value*cos(theta), center_val[1] + radius.value*sin(theta) )
plot( center_val[0], center_val[1], 'x', markersize=10 )

# label
title('Chebyshev Centering')
xlabel('x1')
ylabel('x2')

axis([-1, 1, -1, 1])

show()
