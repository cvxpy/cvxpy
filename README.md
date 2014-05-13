CVXPY [![Build Status](https://travis-ci.org/cvxgrp/cvxpy.png?branch=master)](https://travis-ci.org/cvxgrp/cvxpy)
=====================
**Although this project is similar to and named the same as [CVXPY](https://code.google.com/p/cvxpy/), this version is a total rewrite and is incompatible with the old one.**

What is CVXPY?
---------------------
CVXPY is a Python-embedded modeling language for optimization problems. CVXPY allows you to express your problem in a natural way. It automatically transforms the problem into standard form, calls a solver, and unpacks the results.

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
objective = Minimize(sum_entries(square(A*x - b)))
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

Installation
---------------------
See the installation instructions [here](https://github.com/cvxgrp/cvxpy/wiki/CVXPY-installation-instructions).

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
A = Variable(4, 7)
```

### Constants
CVXPY allows you to use your numeric library of choice to construct problem data. Numeric constants (i.e., scalars, vectors, and matrices) may be combined with CVXPY objects in arbitrary [expressions](#expressions). For instance, if `x` is a CVXPY Variable in the expression `A*x + b`, `A` and `b` could be Numpy ndarrays, Python floats, CVXOPT matrices, etc. `A` and `b` could even be different types.

Currently the following types may be used as constants:
* Python numeric types
* CVXOPT dense matrices
* CVXOPT sparse matrices
* Numpy ndarrays
* Numpy matrices

Support for additional types will be added per request.

### Parameters
Parameters are symbolic representations of constants. Parameters should only be used in special cases. The purpose of Parameters is to change the value of a constant in a problem without reconstructing the entire problem. For example, to efficiently solve `Problem(Minimize(expr1 + gamma*expr2), constraints)` for many different values of `gamma`, make `gamma` a Parameter. See [Parameterized Problems](#parameterized-problems) for an example problem that uses parameters.

Parameters are created using the Parameter class. Parameters are created with fixed dimensions. When you create a parameter you have the option of specifying the sign of the parameter's entries (positive, negative, or unknown). The sign is unknown by default. The sign is used in [DCP convexity analysis](#disciplined-convex-programming-dcp). Parameters can be assigned a constant value any time after they are created. The constant value must have the same dimensions and sign as those specified when the Parameter was created.

```
# Positive scalar parameter.
m = Parameter(sign="positive")

# Column vector parameter with unknown sign (by default).
c = Parameter(5)

# Matrix parameter with negative entries.
G = Parameter(4, 7, sign="negative")

# Assigns a constant value to G.
G.value = -numpy.ones((4, 7))
```

### Expressions
Mathematical expressions are stored in Expression objects. Variable and Parameter are subclasses of Expression. Expression objects are created from constants and other expressions. These elements are combined with arithmetic operators or passed as arguments to [Atoms](#atoms).

```
a = Variable()
x = Variable(5)

# expr is an Expression object after each assignment.
expr = 2*x
expr = expr - a
expr = sum_entries(expr) + norm(x, 2)
```

#### Indexing and Slicing
All non-scalar Expression objects can be indexed using the syntax `expr[i,j]`. The syntax `expr[i]` can be used as a shorthand for `expr[i,0]` when `expr` is a column vector. Similarly, `expr[i]` is shorthand for `expr[0,i]` when `expr` is a row vector.

Non-scalar Expressions can also be sliced into using the standard Python slicing syntax. Thus `expr[i:j:k,r]` selects every kth element in column r of `expr`, starting at row i and ending at row j-1.

#### Transpose
The transpose of any expression can be obtained using the syntax `expr.T`.

### Atoms
Atoms are functions that can be used in expressions. Atoms take Expression objects and constants as arguments and return an Expression object.

CVXPY currently supports the following atoms:
* Matrix to scalar atoms
    * `kl_div(x, y)`, `xlog(x/y) - x + y` for scalar `x` and `y`.
    * `lambda_max(x)`, the maximum eigenvalue of `x`. Constrains `x` to be symmetric.
    * `lambda_min(x)`, the minimum eigenvalue of `x`.
    Constrains `x` to be symmetric.
    * `log_det`, the function `log(det(x))` for a positive semidefinite matrix x.
    * `log_sum_exp(x)`, the function `log(sum(exp(x)))`.
    * `norm(x, [p = 2])`
        * For p = 1, the L1 norm of `x`.
        * For p = 2, the L2 norm of `x` for vector `x` and the spectral norm for matrix `x`.
        * For p = "inf", the Infinity norm of `x`.
        * For p = "nuc", the nuclear norm of `x` (i.e. the sum of the singular values).
        * For p = "fro", the Frobenius norm of `x`.
        * Defaults to p = 2 if no value of p is given.
    * `quad_form(x, P)`, gives `x.T*P*x`. If `x` is non-constant, the real parts of the eigenvalues of `P` must be all non-negative or all non-positive.
    * `quad_over_lin(x,y)`, `x.T*x/y`, where y is a positive scalar.
    * `sum_entries(x)`, sums the entries of the expression.
* Matrix to matrix atoms
    * `hstack(*args)`, the horizontal concatenation of the arguments into a block matrix.
    * `vstack(*args)`, the vertical concatenation of the arguments into a block matrix.
* Elementwise atoms
    * `abs(x)`, the absolute value of each element of `x`.
    *  `entr(x)`, `element*log(element)` for each element of `x`.
    * `exp(x)`, e^element for each element of `x`.
    * `huber(x, M=1)`, the huber function applied to each element of `x`.
    * `inv_pos(x)`, 1/element for each element of `x`.
    * `log(x)`, the natural log of each element of `x`.
    * `max_elemwise(*args)`, the maximum for scalar arguments. Vector and matrix arguments are considered elementwise, i.e., `max([1,2],[-1,3])` returns `[1,3]`.
    * `min_elemwise(*args)`, the minimum for scalar arguments. Vector and matrix arguments are considered elementwise, i.e., `max([1,2],[-1,3])` returns `[-1,2]`.
    * `neg(x)`, `max(-element,0)` for each element of `x`.
    * `pos(x)`, `max(element,0)` for each element of `x`.
    * `sqrt(x)`, the square root of each element of `x`.
    * `square(x)`, the square of each element of `x`.

### Disciplined Convex Programming (DCP)

Expressions must follow the rules of Disciplined Convex Programming (DCP). Following the rules of DCP ensures that any problem you construct is convex. An interactive tutorial on DCP is available at <http://dstanford.edu/>.

DCP assigns a curvature and sign to every expression. The possible curvatures are constant, affine, convex, concave, and unknown. These curvatures have a natural heirarchy. Constant expressions are a kind of affine expression, and affine expressions are both convex and concave. The possible signs are positive (i.e., non-negative), negative (i.e., non-positive), and unknown.

The curvature and sign of Variables, constants, and Parameters are easy to determine. Variables are always affine with unknown sign. Constants and Parameters have constant curvature. The sign of a scalar constant is simply the sign of the constant's numeric value. Matrix constants always have unknown sign. The sign of a Parameter is specified when the Parameter is created (see [Parameters](#parameters)).

#### The DCP Rules

##### The No-Product Rule

You can never multiply two non-constant expressions. Doing so in CVXPY will immediately raise an exception.

##### Curvature Rules

The composition rule determines the curvature of an expression from its sub-expressions. Let `f` be a function applied to the expressions `exp1, exp2, ..., expn`. Then `f(exp1, exp2, ..., expn)` is convex if `f` is a convex function and for each `expi` one of the following conditions holds:

* `f` is non-decreasing in argument i and `expi` is convex
* `f` is non-increasing in argument i and `expi` is concave
* `expi` is affine

If one of the `expi` does not satisfy any of the conditions, the curvature of `f(exp1, exp2, ..., expn)` is unknown. In addition, if all the `expi` are constant, then `f(exp1, exp2, ..., expn)` is constant.

All other DCP rules for determining the curvature of an expression can be derived from the composition rule. For example, if `f` is concave then the composition rule can be applied to `-f`. Arithmetic operators are affine functions, so the composition rule also applies to arithmetic expressions.

#### Sign Rules

For some functions monotonicity (i.e. whether the function is increasing or decreasing in each argument) depends on the sign of the arguments. For example, `square(exp)` is increasing if `exp` is positive and decreasing if `exp` is negative. For this reason DCP tracks the signs of expressions as well as the curvatures.

Each function in cvxpy (i.e. atom or arithmetic operator) has a different rule for determining the sign of the function output from the signs of the arguments. These rules are exhaustive, meaning they capture every case where the sign of the output can be determined from the sign of the inputs. Here is the rule for `+` applied to the scalar expressions `exp1` and `exp2`:

The sign of the expression exp1 + exp2 is
* positive if exp1 and exp2 are both positive
* negative if exp1 and exp2 are both negative
* unknown in all other cases

The rules for other functions are equally straightforward.

#### DCP Methods

To check whether an Expression object follows the DCP rules, use the method `expr.is_dcp()`. [Constraints](#constraints), [Objectives](#objectives), and [Problems](#problems) also have an `is_dcp` method.

The curvature of any Expression object is accessible as `expr.curvature`. Similarly, the sign is accessible as `expr.sign`. For example,

```
x = Variable(2)
x.curvature == 'AFFINE'
x.sign == 'UNKNOWN'

expr = square(x)
expr.curvature == 'CONVEX'
expr.sign == 'POSITIVE'
```

You can also examine the curvature and sign of an expression using the following methods:

* Curvature Methods
    * expr.is_constant()
    * expr.is_affine()
    * expr.is_convex()
    * expr.is_concave()
* Sign Methods
    * expr.is_positive()
    * expr.is_negative()

These methods return whether the expression has the curvature or sign in question. Constant expressions are also considered affine, and affine expressions are considered both convex and concave.

### Constraints
Constraint objects are constructed using `==`, `<=`, and `>=` with Expression objects or constants on the left-hand and right-hand sides.

The lefthand and righthand sides of a constraint are analyzed using the DCP rules to ensure the constraint is convex. Equality constraints must be of the form `affine expression == affine expression`. Inequality constraints must be of the form `convex expression <= concave expression`.

### Objectives
Objective objects are constructed using `Minimize(expression)` or `Maximize(expression)`. Use a constant as an argument to `Minimize` or `Maximize` to create an objective for a feasibility problem.

The target expression of a `Minimize` objective must be convex, while the target of a `Maximize` objective must be concave. Convexity and concavity are determined using the DCP rules.

### Problems
Problem objects are constructed using the form `Problem(objective, constraints)`. Here `objective` is an Objective object, and `constraints` is a list of Constraint objects. The `constraints` argument is optional. The default is an empty list.

The objective for a Problem object `p` is stored in the field `p.objective`, and the constraints list is stored in `p.constraints`. The objective and constraints can be changed after the problem is constructed. For example, `p.constraints[0] = x <= 2` replaces the first constraint with the newly created Constraint object `x <= 2`. Changing the objective or constraints does not require any new computation by the Problem object.

The following code constructs and solves a problem:
```
p = Problem(objective, constraints)
result = p.solve()
```

If the problem is feasible and bounded, `p.solve()` will return the optimal value of the objective. If the problem is infeasible, `p.solve()` will return Inf (-Inf) for minimization (maximization) problems. If the problem is unbounded, `p.solve()` will return -Inf (Inf) for minimization (maximization) problems. Finally, if the solver has an error, `p.solve()` will return None. The result of the most recent call to `p.solve()` is stored in `p.value`.

The field `p.status` stores a string indicating the status of the most recent call to `p.solve()`. The possible statuses are

* `cvxpy.OPTIMAL`, for problems with solutions.
* `cvxpy.INFEASIBLE`, for infeasible problems.
* `cvxpy.UNBOUNDED`, for unbounded problems.
* `cvxpy.UNKNOWN`, for solver error.

Once a problem has been solved, the optimal values of the variables can be read from `variable.value`, where `variable` is a Variable object. The values of the dual variables can be read from `constraint.dual_value`, where `constraint` is a Constraint object.

If the problem had no optimal solution, the values of all the primal and dual variables are `None`.

The value of expressions in the problem can also be read from `expr.value`. For example, consider the portfolio optimization problem below:

```
# Constants:
# mu is the vector of expected returns.
# sigma is the covariance matrix.
# gamma is a Parameter that trades off risk and return.

# Variables:
# x is a vector of stock holdings as fractions of total assets.

expected_return = mu*x
risk = quad_form(x, sigma)

objective = Maximize(expected_return - gamma*risk)
p = Problem(objective, [sum_entries(x) == 1])
result = p.solve()

# The optimal expected return.
print expected_return.value

# The optimal risk.
print risk.value
```

The default solver is [ECOS](http://github.com/ifa-ethz/ecos), though [CVXOPT](http://abel.ee.ucla.edu/cvxopt/) and [SCS](http://github.com/cvxgrp/scs) are used for problems that [ECOS](http://github.com/ifa-ethz/ecos) cannot solve. You can force CVXPY to use a particular solver:

```
p = Problem(objective, constraints)

# Solve with ECOS.
result = p.solve(solver=cvxpy.ECOS)

# Solve with CVXOPT.
result = p.solve(solver=cvxpy.CVXOPT)

# Solve with SCS.
result = p.solve(solver=cvxpy.SCS)
```

To see the full output from the solver, use the `verbose` keyword. The solver output will be printed to the console.

```
p.solve(verbose=True)
```

You can specify solver options for [CVXOPT](http://abel.ee.ucla.edu/cvxopt/) and [SCS](http://github.com/cvxgrp/scs), such as the maximum number of iterations. Create an `opts` dict mapping option keyword to option value and call `p.solve` with the keyword argument `solver_specific_opts=opts`.

Features
=====================

Parameterized Problems
---------------------
Parameters allow you to change value of constants without reconstructing the problem. The following example defines a LASSO problem. The value of gamma is varied to construct a tradeoff curve of the least squares penalty vs. the cardinality of x. The problem instances can be solved efficiently both serially or in parallel.

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
objective = Minimize(sum_entries(square(A*x - b)) + gamma*norm(x, 1))
p = Problem(objective)

# Assign a value to gamma and find the optimal x.
def get_x(gamma_value):
    gamma.value = gamma_value
    result = p.solve()
    return x.value

gammas = np.logspace(-1, 2, num=2)
# Serial computation.
x_values = [get_x(value) for value in gammas]

# Parallel computation.
pool = Pool(processes = 4)
x_values = pool.map(get_x, gammas)

# Construct a trade off curve using the x_values.
...
```

Object Oriented Convex Optimization
---------------------
CVXPY enables an object-oriented approach to constructing optimization problems. An object-oriented approach is simpler and more flexible than the traditional method of constructing problems by embedding information in matrices.

Consider the max-flow problem with N nodes and E edges. We can define the problem explicitly by constructing an N by E incidence matrix A. A[i,j] is +1 if edge j enters node i, -1 if edge j leaves node i, and 0 otherwise. The source and sink are the last two edges. The problem becomes

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

The Edge class exposes the flow into and out of the edge. The capacity constraint is stored locally in the Edge object. The graph structure is also stored locally, by calling `edge.connect(node1, node2)` for each edge.

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

To see the object-oriented approach applied to more complex flow problems, look in the examples/flows/ directory.

Non-Convex Extensions
---------------------
Many non-convex optimization problems can be solved exactly or approximately via a sequence of convex optimization problems. CVXPY can easily be extended to handle such non-convex problems. The examples/mixed_integer package uses the Alternating Direction Method of Multipliers (ADMM) as a heuristic for mixed integer problems.

The following code performs feature selection on a linear kernel SVM classifier using a cardinality constraint:

```
import cvxpy as cp
import mixed_integer as mi
import cvxopt

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

N = 50
M = 40
n = 10
data = []
for i in range(N):
    data += [(1, cvxopt.normal(n, mean=1.0, std=2.0))]
for i in range(M):
    data += [(-1, cvxopt.normal(n, mean=-1.0, std=2.0))]

# Construct problem.
gamma = Parameter(sign="positive")
gamma.value = 0.1
# 'a' is a variable constrained to have at most 6 non-zero entries.
a = mi.SparseVar(n, nonzeros=6)
b = Variable()

slack = [pos(1 - label*(sample.T*a - b)) for (label, sample) in data]
objective = Minimize(norm(a, 2) + gamma*sum(slack))
p = Problem(objective)
# Extensions can attach new solve methods to the CVXPY Problem class.
p.solve(method="admm")

# Count misclassifications.
errors = 0
for label, sample in data:
    if label*(sample.T*a - b).value < 0:
        errors += 1

print "%s misclassifications" % errors
print a.value
print b.value
```

[sphinx]: http://sphinx-doc.org
