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
import numpy as np

# Feature selection on a linear kernel SVM classifier.
# Uses the Alternating Direction Method of Multipliers
# with a (non-convex) cardinality constraint.

# Generate data.
np.random.seed(1)
N = 50
M = 40
n = 10
data = []
for i in xrange(N):
    data += [(1, np.random.normal(1.0, 2.0, (n, 1)))]
for i in xrange(M):
    data += [(-1, np.random.normal(-1.0, 2.0, (n, 1)))]

# Construct problem.
gamma = Parameter(sign="positive")
gamma.value = 0.1
# 'a' is a variable constrained to have at most 6 non-zero entries.
a = Card(n, k=6)
b = Variable()

slack = [pos(1 - label*(sample.T*a - b)) for (label, sample) in data]
objective = Minimize(norm(a, 2) + gamma*sum(slack))
p = Problem(objective)
# Extensions can attach new solve methods to the CVXPY Problem class.
p.solve(method="admm")

# Count misclassifications.
error = 0
for label, sample in data:
    if not label*(a.value.T*sample - b.value)[0] >= 0:
        error += 1

print "%s misclassifications" % error
print a.value
print b.value
