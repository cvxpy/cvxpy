CVXPY
=====================
What is CVXPY?
---------------------
CVXPY is a Python-embedded modeling language for optimization problems. CVXPY lets you express your problem in a natural way. It automatically transforms the problem into standard form, calls a solver, and unpacks the results.

For example, the following code solves a least-squares problem where the variable is constrained by lower and upper bounds:

```
from cvxpy import *
import cvxopt

# Problem data.
m = 30
n = 20
A = cvxopt.normal(m,n)
b = cvxopt.normal(m)

# Construct the problem.
x = Variable(n)
p = Problem( Minimize(sum(square(A*x - b))),
             [0 <= x,
              x <= 1]
)

# The optimal objective is returned by p.solve().
result = p.solve()
# The optimal value for x is stored in x.value.
print x.value
```

The general form for constructing a CVXPY problem is `Problem(objective, constraints)`. The objective is either `Minimize(...)` or `Maximize(...)`. The constraints are a list of expressions of the form `... == ...`, `... <= ...`, or `... >= ...`.

For convex optimization, CVXPY problems must follow the rules of Disciplined Convex Programming (DCP). For an interactive tutorial on DCP, visit <http://dcp.stanford.edu/>.

The available atomic functions are those present in the cvxpy/atoms/ directory.

To see more examples using CVXPY, look in the examples directory. 

Prerequisites
---------------------
CVXPY requires:
* Python 2.7
* [CVXOPT](http://abel.ee.ucla.edu/cvxopt/)
* [ECOS](http://github.com/ifa-ethz/ecos)

To run the unit tests, you additionally need [Nose](http://nose.readthedocs.org).

Installation
---------------------
To install CVXPY, navigate to the top-level directory and call
```
python setup.py install
```
If you have [Nose](http://nose.readthedocs.org) installed, you can verify the CVXPY installation by running
```
nosetests cvxpy/tests/
```

Basic Usage
---------------------
Variables are created using the Variable class.
```
# Scalar variable
a = Variable()

# Column vector variable of length 5.
x = Variable(5)

# Matrix variable with 4 rows and 7 columns.
A = Variable(4,7)
```

Features
=====================

Problem Data
---------------------
CVXPY lets you construct problem data using your library of choice. Certain libraries, such as Numpy, require a lightweight wrapper to support operator overloading. The following code constructs A and b from Numpy ndarrays.

```
from cvxpy import numpy as np

A = np.ndarray(...)
b = np.ndarray(...)
```

Parameters allow you to change the problem data without reconstructing the problem. The following example defines a LASSO problem. The value of gamma is varied to construct a tradeoff curve of the least squares penalty vs. the cardinality of x.

```
from cvxpy import *
from cvxpy import numpy as np
import cvxopt

# Problem data.
n = 10
m = 5
A = cvxopt.normal(n,m)
b = cvxopt.normal(n)
gamma = Parameter(sign="positive")

# Construct the problem.
x = Variable(m)
objective = Minimize(sum(square(A*x - b)) + gamma*norm1(x))
p = Problem(objective)

# Vary gamma for trade-off curve.
x_values = []
for value in np.logspace(-1, 2, num=100):
    gamma.value = value
    p.solve()
    x_values.append(x.value)

# Construct a trade off curve using the x_values.
...
```

Parameterized problems can be solved in parallel. See examples/stock_tradeoff.py for an example.

Object Oriented Optimization
---------------------
CVXPY enables an object oriented approach to constructing optimization problems. An object oriented approach is simpler and more flexible than the traditional method of constructing problems by embedding information in matrices.

Consider the max-flow problem with N nodes and E edges. We can define the problem explicitly by constructing an N by E incidence matrix A. A[i,j] is +1 if edge j enters node i, -1 if edge j leaves node i, and 0 otherwise. The source and sink are the last two edges. The problem becomes:

```
# A is the incidence matrix. c is a vector of edge capacities.
flows = Variable(E-2)
source = Variable()
sink = Variable()
p = Problem(Maximize(source),
            [A*vstack(flows,source,sink) == 0,
             0 <= flows,
             flows <= c])
```

The more natural way to frame the max-flow problem is not in terms of incidence matrices, however, but in terms of the properties of edges and nodes. We can write an Edge class to capture these properties.

```
class Edge(object):
    """ An undirected, capacity limited edge. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.in_flow = Variable()
        self.out_flow = Variable()

    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        in_node.edge_flows.append(self.in_flow)
        out_node.edge_flows.append(self.out_flow)

    # Returns the edge's internal constraints.
    def constraints(self):
        return [self.in_flow + self.out_flow == 0,
                abs(self.in_flow) <= self.capacity]
```

The Edge class exposes the flow into and out of the edge. The constraints linking the flow in and out and the flows with the capacity are stored locally in the Edge object. The graph structure is also stored locally, by calling `edge.connect(node1, node2)` for each edge.

We also define a Node class:

```
class Node(object):
    """ A node with a target flow accumulation. """
    def __init__(self, accumulation=0):
        self.accumulation = accumulation
        self.edge_flows = []
    
    def constraints(self):
        return [sum(f for f in self.edge_flows) == self.accumulation]
```

Nodes have a target amount of flow to accumulate. Sources and sinks are Nodes with a variable as their accumulation target.

Suppose `nodes` is a list of all the nodes, `edges` is a list of all the edges, and `sink` is the sink node. The problem becomes:

```
constraints = []
for obj in nodes + edges:
    constraints += obj.constraints()
p = Problem(Maximize(sink.accumulation), constraints)
```

Note that the problem has been reframed from maximizing the flow along the source edge to maximizing the accumulation at the sink node. We could easily extend the Edge and Node class to model an electrical grid. Sink nodes would be consumers. Source nodes would be power stations, which generate electricity at a cost. A node could be both a source and a sink, which would represent energy storage facilities or a consumer who contributes to the grid. We could add energy loss along edges to more accurately model transmission lines. The entire grid construct could be embedded in a time series model.

To see the object oriented approach to flow problems fleshed out in more detail, look in the examples/flows/ directory.

Non-Convex Extensions
---------------------
Many non-convex optimization problems can be solved exactly or approximately via a sequence of convex optimization problems. CVXPY can easily be extended to handle such non-convex problems. The examples/mixed_integer package uses the Alternating Direction Method of Multipliers (ADMM) as a heuristic for mixed integer problems.

The following code performs feature selection on a linear kernel SVM classifier using a cardinality constraint:

```
from cvxpy import *
from mixed_integer import *
import cvxopt

# Generate data.
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
```