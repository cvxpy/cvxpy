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
### Variables
Variables are created using the Variable class.
```
# Scalar variable.
a = Variable()

# Column vector variable of length 5.
x = Variable(5)

# Matrix variable with 4 rows and 7 columns.
A = Variable(4,7)
```

### Constants
The following types may be used as constants:
* Python numeric types
* CVXOPT dense matrices
* CVXOPT sparse matrices
* Numpy ndarrays
* Numpy matrices

Support for additional types will be added per request. See [Problem Data](#problem-data) for more information on using numeric libraries with CVXPY.

### Parameters
Parameters are symbolic representations of constants. Parameters have fixed dimensions. The sign of a parameter's entries is also fixed as positive, negative, or unknown. Parameters are created using the Parameter class. Parameters can be assigned a constant value any time after they are created.

```
# Positive scalar parameter.
m = Parameter(sign="positive")

# Column vector parameter with unknown sign (by default).
c = Parameter(5)

# Matrix parameter with negative entries.
G = Parameter(4,7,sign="negative")

# Assigns a constant value to G.
G.value = cvxopt.matrix(...)
```

### Expressions
Mathematical expressions are stored in Expression objects. Variable and Parameter are subclasses of Expression. Expression objects are created from constants and other expressions. These elements are combined with arithmetic operators or passed as arguments to [Atoms](#atoms).

```
a = Variable()
x = Variable(5)

# exp is an Expression object after each assignment.
exp = 2*x
exp = exp - a
exp = sum(exp) + norm2(x)
```

Expressions must follow the rules of Disciplined Convex Programming (DCP). An interactive tutorial on DCP is available at <http://dcp.stanford.edu/>.

### Indexing and Iteration
All Expression objects can be indexed using the syntax `exp[i,j]` if `exp` is a matrix and `exp[i]` if exp is a vector.

Expressions are also iterable. Iterating over an expression returns indexes into the expression in column-major order. Thus if `exp` is a 2 by 2 matrix, `[elem for elem in exp]` evaluates to `[exp[0,0], exp[1,0], exp[0,1], exp[1,1]]`. The built-in Python `sum` can be used on expressions because of the support for iteration.

### Atoms
Atoms are functions that can be used in expressions. Atoms take Expression objects and constants as arguments and return an Expression object. 

CVXPY currently supports the following atoms:
* Atoms that return scalars
    * `norm1(x)`, the L1 norm of `x`.
    * `norm2(x)`, the L2 norm of `x`.
    * `normInf(x)`, the Infinity norm of `x`.
    * `quad_over_lin(x,y)`, x'*x/y, where y is a positive scalar.
* Elementwise atoms
    * `abs(x)`, the absolute value of each element of `x`.
    * `inv_pos(x)`, 1/element for each element of `x`.
    * `min(x)`, the absolute value of each element of `x`.
    * `pos(x)`, `max(element,0)` for each element of `x`.
    * `sqrt(x)`, the square root of each element of `x`.
    * `square(x)`, the square of each element of `x`.
* Variable argument atoms
    * `max(x,y,...)`, the maximum for scalar arguments. Vector and matrix arguments are considered elementwise, i.e. `max([1,2],[-1,3])` returns `[1,3]`.
    * `min(x,y,...)`, the minimum for scalar arguments. Vector and matrix arguments are considered elementwise, i.e. `max([1,2],[-1,3])` returns `[-1,2]`. 
    * `vstack(x,y,...)`, the vertical concatenation of the arguments into a block matrix.

### Constraints
Constraint objects are constructed using `==`, `<=`, and `>=` with Expression objects or constants on the left-hand and right-hand sides.

### Objectives
Objective objects are constructed using `Minimize(expression)` or `Maximize(expression)`. Use a constant as an argument to `Minimize` or `Maximize` to create a feasibility objective.

### Problems
Problem objects are constructed using the form `Problem(objective, constraints)`. Here `objective` is an Objective object, and `constraints` is a list of Constraint objects. The `constraints` argument is optional. The default is an empty list.

The objective for a Problem object `p` is stored in the field `p.objective`, and the constraints list is stored in `p.constraints`. The objective and constraints can be changed after the problem is constructed. For example, `p.constraints[0] = x <= 2` replaces the first constraint with the newly created Constraint object `x <= 2`. Changing the objective or constraints does not require any new computation by the Problem object.

The following code constructs and solves a problem:
```
p = Problem(objective, constraints)
result = p.solve()
```

If the problem is feasible and bounded, `result` will hold the optimal value of the objective. If the problem is unfeasible or unbounded, `result` will hold the constant `cvxpy.INFEASIBLE` or `cvxpy.UNBOUNDED`, respectively.

Once a problem has been solved, the optimal values of the variables can be read from `variable.value`. The values of the dual variables can be read from `constraint.dual_value`.


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