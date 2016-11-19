# Incidence matrix approach.
from cvxpy import *
import create_graph as g
import pickle
import cvxopt

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
            [A*vstack(flows,source,sink) == 0,
             0 <= flows,
             flows <= c])
result = p.solve()
print result
