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
# Section 6.1.2: Residual minimization with deadzone penalty
# Ported from cvx matlab to cvxpy by Misrab Faizullah-Khan
# Original comments below

# Boyd & Vandenberghe "Convex Optimization"
# Joelle Skaf - 08/17/05
#
# The penalty function approximation problem has the form:
#               minimize    sum(deadzone(Ax - b))
# where 'deadzone' is the deadzone penalty function
#               deadzone(y) = max(abs(y)-1,0)

# Input data
m = 16
n = 8
A = cvxopt.normal(m,n)
b = cvxopt.normal(m,1)

# Formulate the problem
x = Variable(n)
objective = Minimize( sum(maximum( abs(A*x -b) - 1 , 0 )) )
p = Problem(objective, [])

# Solve it
print ('Computing the optimal solution of the deadzone approximation problem:')
p.solve()

print ('Optimal vector:')
print (x.value)

print ('Residual vector:')
print (A*x.value - b)

