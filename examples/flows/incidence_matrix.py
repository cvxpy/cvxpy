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
import pickle

import create_graph as g
import cvxopt

from cvxpy import Maximize, Problem, Variable, vstack

# Read a graph from a file.
f = open(g.FILE, 'r')
data = pickle.load(f)
f.close()

# Construct incidence matrix and capacities vector.
node_count = data[g.NODE_COUNT_KEY]
edges = data[g.EDGES_KEY]
E = 2*len(edges)
A = cvxopt.matrix(0,(node_count, E+2), tc='d')
c = cvxopt.matrix(1000,(E,1), tc='d')
for i,(n1,n2,capacity) in enumerate(edges):
    A[n1,2*i] = -1
    A[n2,2*i] = 1
    A[n1,2*i+1] = 1
    A[n2,2*i+1] = -1
    c[2*i] = capacity
    c[2*i+1] = capacity
# Add source.
A[0,E] = 1
# Add sink.
A[-1,E+1] = -1
# Construct the problem.
flows = Variable(E)
source = Variable()
sink = Variable()
p = Problem(Maximize(source),
            [A*vstack([flows,source,sink]) == 0,
             0 <= flows,
             flows <= c])
result = p.solve()
print(result)
