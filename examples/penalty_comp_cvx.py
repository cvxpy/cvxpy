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
import sys

import cvxopt
import numpy as np
from pylab import *
import math

from cvxpy import *

# Taken from CVX website http://cvxr.com/cvx/examples/
# Figure 6.2: Penalty function approximation
# Ported from cvx matlab to cvxpy by Misrab Faizullah-Khan
# Original comments below

# Section 6.1.2
# Boyd & Vandenberghe "Convex Optimization"
# Original by Lieven Vandenberghe
# Adapted for CVX Argyris Zymnis - 10/2005
#
# Comparison of the ell1, ell2, deadzone-linear and log-barrier
# penalty functions for the approximation problem:
#       minimize phi(A*x-b),
#
# where phi(x) is the penalty function

# Generate input data
m, n = 100, 30
A = cvxopt.normal(m,n) #np.random.randn(m,n)
b = cvxopt.normal(m,1) #np.random.randn(m,1)

# l-1 approximation
x1 = Variable(n)
objective1 = Minimize( norm(A*x1-b, 1) )
p1 = Problem(objective1, [])
#p1 = Problem(Minimize( norm(A*x1-b, 1), []))

# l-2 approximation
x2 = Variable(n)
objective2 = Minimize( norm(A*x2-b, 2) )
p2 = Problem(objective2, [])

# deadzone approximation
# minimize sum(deadzone(Ax+b,0.5))
# deadzone(y,z) = max(abs(y)-z,0)
def deadzone(y,z):
	return pos(abs(y)-z)

dz = 0.5
xdz = Variable(n)
objective3 = Minimize( sum( deadzone(A*xdz+b, dz) ) )
p3 = Problem(objective3, [])

# Solve the problems
p1.solve()
p2.solve()
p3.solve()

# Plot histogram of residuals
range_max=2.0
#rr = np.arange(-range_max, range_max, 1e-2)
rr = np.linspace(-2, 3, 20)



# l-1 plot
subplot(3, 1, 1)
n, bins, patches = hist(A*x1.value-b, 50, range=[-2, 2])
plot(bins, np.abs(bins)*35/3, '-') # multiply by scaling factor for plot
ylabel('l-1 norm')
title('Penalty function approximation')

# l-2 plot
subplot(3, 1, 2)
n, bins, patches = hist(A*x2.value-b, 50,  range=[-2, 2])
plot(bins, np.power(bins, 2)*2, '-')
ylabel('l-2 norm')

# deadzone plot
subplot(3, 1, 3)
n, bins, patches = hist(A*xdz.value+b, 50, range=[-2, 2])
zeros = np.array([0 for x in bins])
plot(bins, np.maximum((np.abs(bins)-dz)*35/3, zeros), '-')
ylabel('deadzone')

show()
