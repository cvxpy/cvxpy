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

import numpy as np
from mixed_integer import *

from cvxpy import Card, Minimize, Parameter, Problem, Variable, norm, pos

# Feature selection on a linear kernel SVM classifier.
# Uses the Alternating Direction Method of Multipliers
# with a (non-convex) cardinality constraint.

# Generate data.
np.random.seed(1)
N = 50
M = 40
n = 10
data = []
for i in range(N):
    data += [(1, np.random.normal(1.0, 2.0, (n, 1)))]
for i in range(M):
    data += [(-1, np.random.normal(-1.0, 2.0, (n, 1)))]

# Construct problem.
gamma = Parameter(nonneg=True)
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

print("%s misclassifications" % error)
print(a.value)
print(b.value)
