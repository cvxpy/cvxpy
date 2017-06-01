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

# for decimal division
from __future__ import division

import cvxopt
import numpy as np
from pylab import *
import math

from cvxpy import *

# Taken from CVX website http://cvxr.com/cvx/examples/
# Example: Section 5.2.5: Mixed strategies for matrix games (LP formulation)
# Ported from cvx matlab to cvxpy by Misrab Faizullah-Khan
# Original comments below


# Boyd & Vandenberghe, "Convex Optimization"
# Joelle Skaf - 08/24/05
#
# Player 1 wishes to choose u to minimize his expected payoff u'Pv, while
# player 2 wishes to choose v to maximize u'Pv, where P is the payoff
# matrix, u and v are the probability distributions of the choices of each
# player (i.e. u>=0, v>=0, sum(u_i)=1, sum(v_i)=1)
# LP formulation:   minimize    t
#                       s.t.    u >=0 , sum(u) = 1, P'*u <= t*1
#                   maximize    t
#                       s.t.    v >=0 , sum(v) = 1, P*v >= t*1

# Input data
n = 12
m = 12
P = cvxopt.normal(n,m)

# Variables for two players
x = Variable(n)
y = Variable(m)
t1 = Variable()
t2 = Variable()

# Note in one case we are maximizing; in the other we are minimizing
objective1 = Minimize(t1)
objective2 = Maximize(t2)

constraints1 = [ x>=0, sum_entries(x)==1, P.T*x <= t1 ]
constraints2 = [ y>=0, sum_entries(y)==1, P*y >= t2 ]


p1 = Problem(objective1, constraints1)
p2 = Problem(objective2, constraints2)

# Optimal strategy for Player 1
print 'Computing the optimal strategy for player 1 ... '
result1 = p1.solve()
print 'Done!'

# Optimal strategy for Player 2
print 'Computing the optimal strategy for player 2 ... '
result2 = p2.solve()
print 'Done!'

# Displaying results
print '------------------------------------------------------------------------'
print 'The optimal strategies for players 1 and 2 are respectively: '
print x.value, y.value
print 'The expected payoffs for player 1 and player 2 respectively are: '
print result1, result2
print 'They are equal as expected!'
## ISSUE: THEY AREN'T EXACTLY EQUAL FOR SOME REASON!
