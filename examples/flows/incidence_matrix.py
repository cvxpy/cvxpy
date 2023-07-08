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

# Incidence matrix approach.
import numpy as np
import pickle

import create_graph as g

from cvxpy import Maximize, Parameter, Problem, Variable, vstack

# Read a graph from a file.
f = open(g.FILE, 'rb')
data = pickle.load(f)
f.close()

# Construct incidence matrix and capacities vector.
node_count = data[g.NODE_COUNT_KEY]
edges = data[g.EDGES_KEY]
E = 2 * len(edges)

A = Parameter((node_count, E + 2))
A.value = np.zeros((node_count, E + 2))
c = Parameter((E))
c.value = np.full((E), 1000)
for i, (n1, n2, capacity) in enumerate(edges):
    A.value[n1, 2 * i] = -1
    A.value[n2, 2 * i] = 1
    A.value[n1, 2 * i + 1] = 1
    A.value[n2, 2 * i + 1] = -1
    c.value[2 * i] = capacity
    c.value[2 * i + 1] = capacity
# Add source.
A.value[0, E] = 1
# Add sink.
A.value[-1, E + 1] = -1
# Construct the problem.
flows = Variable(E)
source = Variable()
sink = Variable()
p = Problem(Maximize(source),
            [A @ vstack([f for f in flows] + [source, sink]) == 0,
             0 <= flows,
             flows <= c])
result = p.solve()
print(result)
