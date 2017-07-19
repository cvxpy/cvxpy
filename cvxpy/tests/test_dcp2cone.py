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

import cvxpy as cvx
import numpy.random as r
import numpy as np
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone


n = 5
x = cvx.Variable(n)
y = cvx.Variable()
A = r.rand(n,n)
b = r.rand(n,1)

l = np.random.randn(5, 4)
c = [cvx.abs(A*x + b) <= 2, cvx.abs(y) + x[0] <= 1, cvx.log1p(x) >= 5]
c.append(cvx.log_sum_exp(l, axis=0) <= 10)
X = cvx.Variable((5, 5))
c.append(cvx.log_det(X) >= 10)
cvx.Minimize(x[0])
prob = cvx.Problem(cvx.Minimize(x[0] + x[1] + y), c)

d2c = Dcp2Cone()
d2c.accepts(prob)
new_prob = d2c.apply(prob)

print(prob)
print('\n\n')
print(new_prob)

prob.solve()
new_prob.solve()

print(prob.value)
print(new_prob.value)
