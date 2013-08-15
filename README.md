CVXPY
=====================
What is CVXPY?
---------------------
CVXPY is a Python-embedded modeling language for optimization problems. CVXPY lets you express your problem in a natural way. It automatically transforms the problem into standard form, calls a solver, and unpacks the results.

The following code expresses a linear program in CVXPY:
```
from cvxpy import *
import cvxopt

p = Problem( Minimize(c.T*) )

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

To run the unit tests, you additionally need
* [Nose](http://nose.readthedocs.org)

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

Features
=====================

Problem Data
---------------------
CVXPY lets you construct problem data using your library of choice. Certain libraries, such as Numpy, require a lightweight wrapper to support operator overloading. The following code constructs A and b from Numpy ndarrays.

```
import cvxpy.numpy as np

A = np.ndarray(...)
b = np.ndarray(...)
```

You can solve the same problem with different problem data using parameters. The value of a parameter can be initialized and changed after the problem is constructed. The following example constructs a tradeoff curve for a LASSO problem of the least squares penalty vs. the cardinality of x.

```
import cvxpy.numpy as np

# Problem data.
n = 10
m = 5
A = cvxopt.normal(n,m)
b = cvxopt.normal(m)
lambda = Parameter("positive")

# Construct the problem.
x = Variable(m)
objective = Minimize(sum(square(A*x - b)) + lambda*norm1(x))
p = Problem(objective)

# Vary lambda for trade-off curve.
x_values = []
for value in np.logspace(-1, 2, num=100):
    lambda.value = value
    p.solve()
    x_values.append(x.value)

# Construct a trade off curve using the x_values.
...
```

Parameterized problems can be solved in parallel. See examples/stock_tradeoff.py for an example.

Object Oriented Optimization
---------------------
Since CVXPY is embedded in Python, you have all the expressive power of a modern object oriented language in constructing CVXPY optimization problems. The following section shows the distinction between the classical approach to constructing optimization problems, which focuses on embedding information in matrices, and the object oriented approach possible with CVXPY.

Consider the max-flow problem with N nodes and E edges. You can define the problem explicitly by constructing an N by E incidence matrix A. A[i,j] is +1 if edge j enters node i, -1 if edge j leaves node i, and 0 otherwise. The source and sink are the last two edges. The problem becomes:

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

The more natural way to frame the max-flow problem is not in terms of incidence matrices, however, but in terms of the properties of edges and nodes. We can write an Edge class to capture those properties.

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

The Edge class exposes the two ends of the edge. The constraints linking the two ends and the Edge capacity are stored locally in the Edge object. The graph structure is also stored locally, by calling `edge.connect(node1, node2)` for each edge.

We also define a Node class:

class Node(object):
    """ A node with a target flow accumulation. """
    def __init__(self, accumulation=0):
        self.accumulation = accumulation
        self.edge_flows = []
    
    def constraints(self):
        return [sum(f for f in self.edge_flows) == self.accumulation]

Nodes have a target amount of flow to accumulate. Sources and sinks are Nodes with a variable as their accumulation target.

Suppose "nodes" is a list of all the nodes, "edges" is a list of all the edges, and "sink" is the sink node. The problem becomes:

```
constraints = []
for obj in nodes + edges:
    constraints += obj.constraints()
p = Problem(Maximize(sink.accumulation), constraints)
```

Note that the problem has been reframed from maximizing the flow along the source edge to maximizing the accumulation at the sink node. We could easily extend the Edge and Node class to model an electrical grid. Sink nodes would be consumers. Source nodes would be power stations, which generate electricity at a cost. A node could be both a source and a sink, which would represent energy storage facilities or a consumer with solar panels. We could add energy loss along edges. The entire grid construct could be embedded in a time series model.