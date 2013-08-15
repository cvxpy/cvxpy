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

from cvxpy import *
from mixed_integer import *
import cvxopt

# SVM with feature selection using cardinality constraints.
# Generate data.
cvxopt.setseed(2)
N = 50
M = 40
n = 10
data = []
map(data.append, ( (1,cvxopt.normal(n, mean=1.0, std=2.0)) for i in range(N) ))
map(data.append, ( (-1,cvxopt.normal(n, mean=-1.0, std=2.0)) for i in range(M) ))

# Construct problem.
gamma = Parameter(sign="positive")
gamma.value = 0.1
a = Variable(n)
b = Variable()

slack = (pos(1-label*(sample.T*a-b)) for (label,sample) in data)
objective = Minimize(norm2(a) + gamma*sum(slack))
p = Problem(objective, [card(n,k=6) == a])
p.solve(method="admm")

# Count misclassifications.
error = 0
for label,sample in data:
    if not label*(a.value.T*sample - b.value)[0] >= 0:
        error += 1

print "%s misclassifications" % error
print a.value
print b.value