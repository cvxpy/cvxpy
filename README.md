CVXPY [![Build Status](https://travis-ci.org/cvxgrp/cvxpy.png)](https://travis-ci.org/cvxgrp/cvxpy)
=====================
**Although this project is similar to and named the same as [CVXPY](http://www.stanford.edu/~ttinoco/cvxpy/), this version is a total rewrite and is incompatible with the now-deprecated CVXPY from Stanford.**

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
objective = Minimize(sum(square(A*x - b)))
constraints = [0 <= x, x <= 1]
p = Problem(objective, constraints)

# The optimal objective is returned by p.solve().
result = p.solve()
# The optimal value for x is stored in x.value.
print x.value
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print constraints[0].dual_value
```

Prerequisites
---------------------
CVXPY requires:
* Python 2.7
* [CVXOPT](http://abel.ee.ucla.edu/cvxopt/)
* [ECOS](http://github.com/ifa-ethz/ecos)
* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)

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
CVXPY allows you to use your numeric library of choice to construct problem data. Numeric constants (i.e. scalars, vectors, and matrices) may be combined with CVXPY objects in arbitrary [expressions](#expressions). For instance, if `x` is a CVXPY Variable in the expression `A*x + b`, `A` and `b` could be Numpy ndarrays, Python floats, CVXOPT matrices, etc. `A` and `b` could even be different types.

Currently the following types may be used as constants:
* Python numeric types
* CVXOPT dense matrices
* CVXOPT sparse matrices
* Numpy ndarrays (see [Problem Data](#problem-data))
* Numpy matrices (see [Problem Data](#problem-data))

Support for additional types will be added per request. See [Problem Data](#problem-data) for more information on using numeric libraries with CVXPY.

### Parameters
Parameters are symbolic representations of constants. Parameters should only be used in special cases. The purpose of Parameters is to change the value of a constant in a problem without reconstructing the entire problem. For example, to efficiently solve `Problem(Minimize(expr1 + gamma*expr2), constraints)` for many different values of `gamma`, make `gamma` a Parameter. See [Problem Data](#problem-data) for an example problem that uses parameters.

Parameters are created using the Parameter class. Parameters are created with fixed dimensions. When creating a parameter, there is also the option of specifying the sign of the parameter's entries (positive, negative, or unknown). The sign is unknown by default. The sign is used in [DCP convexity analysis](#disciplined-convex-programming-dcp). Parameters can be assigned a constant value any time after they are created.

```
# Positive scalar parameter.
m = Parameter(sign="positive")

# Column vector parameter with unknown sign (by default).
c = Parameter(5)

# Matrix parameter with negative entries.
G = Parameter(4,7,sign="negative")

# Assigns a constant value to G.
G.value = -numpy.ones((4,7))
```

### Expressions
Mathematical expressions are stored in Expression objects. Variable and Parameter are subclasses of Expression. Expression objects are created from constants and other expressions. These elements are combined with arithmetic operators or passed as arguments to [Atoms](#atoms).

```
a = Variable()
x = Variable(5)

# expr is an Expression object after each assignment.
expr = 2*x
expr = expr - a
expr = sum(expr) + norm2(x)
```

#### Indexing and Slicing
All non-scalar Expression objects can be indexed using the syntax `expr[i,j]`. The syntax `expr[i]` can be used as a shorthand for `expr[i,0]` when `expr` is a column vector. Similarly, `expr[i]` is shorthand for `expr[0,i]` when `expr` is a row vector.

Non-scalar Expressions can also be sliced into using the standard Python slicing syntax. Thus `expr[i:j:k,r]` selects every kth element in column r of `expr`, starting at row i and ending at row j-1.

#### Iteration
Expressions are iterable. Iterating over an expression returns indices into the expression in column-major order. Thus if `expr` is a 2 by 2 matrix, `[elem for elem in expr]` evaluates to `[expr[0,0], expr[1,0], expr[0,1], expr[1,1]]`. The built-in Python `sum` can be used on expressions because of the support for iteration.

#### Transpose
The transpose of any expression can be obtained using the syntax `expr.T`.

### Atoms
Atoms are functions that can be used in expressions. Atoms take Expression objects and constants as arguments and return an Expression object. 

CVXPY currently supports the following atoms:
* Vector to scalar atoms
    * `norm1(x)`, the L1 norm of `x`.
    * `norm2(x)`, the L2 norm of `x`.
    * `normInf(x)`, the Infinity norm of `x`.
    * `quad_over_lin(x,y)`, x'*x/y, where y is a positive scalar.
* Matrix to scalar atoms
    * `sigma_max(X)`, the maximum singular value of `X`.
* Matrix to matrix atoms
    * `max(*args)`, the maximum for scalar arguments. Vector and matrix arguments are considered elementwise, i.e. `max([1,2],[-1,3])` returns `[1,3]`.
    * `min(*args)`, the minimum for scalar arguments. Vector and matrix arguments are considered elementwise, i.e. `max([1,2],[-1,3])` returns `[-1,2]`. 
    * `vstack(*args)`, the vertical concatenation of the arguments into a block matrix.
* Elementwise atoms
    * `abs(x)`, the absolute value of each element of `x`.
    * `inv_pos(x)`, 1/element for each element of `x`.
    * `log(x)`, the natural log of each element of `x`.
    * `neg(x)`, `max(-element,0)` for each element of `x`.
    * `pos(x)`, `max(element,0)` for each element of `x`.
    * `sqrt(x)`, the square root of each element of `x`.
    * `square(x)`, the square of each element of `x`.

### Disciplined Convex Programming (DCP)

Expressions must follow the rules of Disciplined Convex Programming (DCP). Following the rules of DCP ensures that any problem you construct is convex. An interactive tutorial on DCP is available at <http://dcp.stanford.edu/>.

DCP assigns a curvature and sign to every scalar expression and every element of a matrix expression. The possible curvatures are constant, affine, convex, concave, and unknown. These curvatures have a natural heirarchy. Constant expressions are a kind of affine expression, and affine expressions are both convex and concave. The possible signs are positive (i.e. non-negative), negative (i.e. non-positive), and unknown.

The curvature and sign of Variables, constants, and Parameters are easy to determine. Variables are always affine with unknown sign. Constants and Parameters have constant curvature. The sign of a scalar constant is simply the sign of the constant's numeric value. For matrix constants, a sign is determined for each entry. The sign of a Parameter is specified when the Parameter is created (see [Parameters](#parameters)).

#### The DCP Rules

##### The No-Product Rule

You can never multiply two non-constant expressions. Doing so in cvxpy will immediately raise an exception.

##### Curvature Rules

The curvature composition rule explains how the curvature of an expression is determined from its sub-expressions. Let `f` be a function applied to the expressions `exp1, exp2, ..., expn`. Then `f(exp1, exp2, ..., expn)` is convex if `f` is a convex function and for each `expi` one of the following conditions holds:

* `f` is non-decreasing in argument i and `expi` is convex
* `f` is non-increasing in argument i and `expi` is concave
* `expi` is affine

If one of the `expi` does not satisfy any of the conditions, the curvature of `f(exp1, exp2, ..., expn)` is unknown.

All other DCP rules for determining the curvature of an expression can be derived from the curvature composition rule. For example, if `f` is concave then the curvature composition rule can be applied to `-f`. Arithmetic operators are affine functions, so the curvature composition rule also applies to arithmetic expressions.

#### Sign Rules

For some functions monotonicity (i.e. whether the function is increasing or decreasing in each argument) depends on the sign of the arguments. For example, `square(exp)` is increasing if `exp` is positive and decreasing if `exp` is negative. For this reason DCP tracks the signs of expressions as well as the curvatures.

Each function in cvxpy (i.e. atom or arithmetic operator) has a different rule for determining the sign of the function output from the signs of the arguments. These rules are exhaustive, meaning they capture every case where the sign of the output can be determined from the sign of the inputs. Here is the rule for `+` applied to the scalar expressions `exp1` and `exp2`:

```
The sign of the expression exp1 + exp2 is
* positive if exp1 and exp2 are both positive
* negative if exp1 and exp2 are both negative
* unknown in all other cases
```

The rules for other functions are equally straightforward.

#### DCP Methods

To check whether an Expression object follows the DCP rules, use the method `expr.is_dcp()`. An `is_dcp` method also exists for Constraints, Objectives, and Problems.

The curvature of any Expression object is accessible as `expr.curvature`. Similarly, the sign is accessible as `expr.sign`. The curvature and sign are complex objects. The simplest way to examine the curvature and sign is to use the following methods:

* Curvature Methods
    * expr.curvature.is_constant()
    * expr.curvature.is_affine()
    * expr.curvature.is_convex()
    * expr.curvature.is_concave()
* Sign Methods
    * expr.sign.is_positive()
    * expr.sign.is_negative()

For scalar expressions, these methods return whether the expression has the curvature or sign in question. Constant expressions are also considered affine, and affine expressions are considered both convex and concave.

For matrix expressions, these methods return true only if the method returns true for every entry.

### Constraints
Constraint objects are constructed using `==`, `<=`, and `>=` with Expression objects or constants on the left-hand and right-hand sides.

The lefthand and righthand sides of a constraint are analyzed using the DCP rules to ensure the constraint is convex. Equality constraints must be of the form `affine expression == affine expression`. Inequality constraints must be of the form `convex expression <= concave expression`.

### Objectives
Objective objects are constructed using `Minimize(expression)` or `Maximize(expression)`. Use a constant as an argument to `Minimize` or `Maximize` to create an objective for a feasibility problem.

The target Expression in an objective is analyzed using the DCP rules to ensure the objective is convex. The target of a `Minimize` objective must be convex, while the target of a `Maximize` objective must be concave.

### Problems
Problem objects are constructed using the form `Problem(objective, constraints)`. Here `objective` is an Objective object, and `constraints` is a list of Constraint objects. The `constraints` argument is optional. The default is an empty list.

The objective for a Problem object `p` is stored in the field `p.objective`, and the constraints list is stored in `p.constraints`. The objective and constraints can be changed after the problem is constructed. For example, `p.constraints[0] = x <= 2` replaces the first constraint with the newly created Constraint object `x <= 2`. Changing the objective or constraints does not require any new computation by the Problem object.

The following code constructs and solves a problem:
```
p = Problem(objective, constraints)
result = p.solve()
```

If the problem is feasible and bounded, `p.solve()` will return the optimal value of the objective. If the problem is unfeasible or unbounded, `p.solve()` will return the constant `cvxpy.INFEASIBLE` or `cvxpy.UNBOUNDED`, respectively. Finally, if the solver fails to return a definite result, `p.solve()` will return `cvxpy.UNKNOWN`. 

Use the method `cvxpy.get_status(result)` to convert the result of `p.solve()` to a constant indicating the status. The status will be `cvxpy.SOLVED` if the result is a number and the result itself otherwise.

Once a problem has been solved, the optimal values of the variables can be read from `variable.value`, where `variable` is a Variable object. The values of the dual variables can be read from `constraint.dual_value`, where `constraint` is a Constraint object.

The default solver is [ECOS](http://github.com/ifa-ethz/ecos), though [CVXOPT](http://abel.ee.ucla.edu/cvxopt/) is used for problems that [ECOS](http://github.com/ifa-ethz/ecos) cannot solve. You can force CVXPY to use a particular solver:

```
p = Problem(objective, constraints)

# Solve with ECOS.
result = p.solve(solver=cvxpy.ECOS)

# Solve with CVXOPT.
result = p.solve(solver=cvxpy.CVXOPT)
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

Parameters allow you to change the problem data without reconstructing the problem. The following example defines a LASSO problem. The value of gamma is varied to construct a tradeoff curve of the least squares penalty vs. the cardinality of x. The problem instances can be solved efficiently both serially or in parallel.

```
from cvxpy import *
import numpy as np
import cvxopt
from multiprocessing import Pool

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

# Assign a value to gamma and find the optimal x.
def get_x(gamma_value):
    gamma.value = gamma_value
    result = p.solve()
    return x.value

gammas = np.logspace(-1, 2, num=100)
# Serial computation.
x_values = [get_x(value) for value in gammas]

# Parallel computation.
pool = Pool(processes = 4)
x_values = pool.map(get_x, gammas)

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
        self.flow = Variable()

    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.flow)

    # Returns the edge's internal constraints.
    def constraints(self):
        return [abs(self.flow) <= self.capacity]
```

The Edge class exposes the flow into and out of the edge. The constraints linking the flow in and out and the flows with the capacity are stored locally in the Edge object. The graph structure is also stored locally, by calling `edge.connect(node1, node2)` for each edge.

We also define a Node class:

```
class Node(object):
    """ A node with accumulation. """
    def __init__(self, accumulation=0):
        self.accumulation = accumulation
        self.edge_flows = []
    
    # Returns the node's internal constraints.
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
for i in range(N):
    data += [(1,cvxopt.normal(n, mean=1.0, std=2.0))]
for i in range(M):
    data += [(-1,cvxopt.normal(n, mean=-1.0, std=2.0))]

# Construct problem.
gamma = Parameter(sign="positive")
gamma.value = 0.1
# 'a' is a variable constrained to have at most 6 non-zero entries.
a = SparseVar(n,nonzeros=6)
b = Variable()

slack = [pos(1 - label*(sample.T*a - b)) for (label,sample) in data]
objective = Minimize(norm2(a) + gamma*sum(slack))
p = Problem(objective)
# Extensions can attach new solve methods to the CVXPY Problem class. 
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
