
DMCP
====
A multi-convex optimization problem is one in which the variables can be partitioned into sets over each of which the problem is convex when the other variables are fixed.
It is generally a nonconvex problem.
DMCP package provides a method to verify multi-convexity and find minimal sets of variables that have to be fixed for a problem to be convex, as well as an organized heuristic for multi-convex programming.
The full details of our approach are discussed in [the associated paper]. DMCP is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

Installation
------------
You should first install [CVXPY](http://ww.cvxpy.org/), following the instructions [here](http://www.cvxpy.org/en/latest/install/index.html).

DMCP rules
----------

```
minimize    f_0(x) 
subject to  f_i(x) <= 0, i = 1,..., m
            g_i(x) = 0, i = 1,..., p,
```
where variable ``x`` admits a partition of blocks of variables ``x = (x_1,...,x_N)``, and functions ``f_i`` for ``i = 0,...,m`` and ``g_i`` for ``i = 1,...,p`` are proper.
The problem can be specified as disciplined multi-convex programming (DMCP), if there are index sets ``F_1,..., F_K``, such that their intersection is empty, and for every ``k`` the problem with variables ``x_i`` for all ``i`` in set ``F_k`` fixed to any value can be specified as DCP.

Example
-------
The following code uses DMCP to approximately solve a simple multi-convex problem.
```
x_1 = Variable(1)
x_2 = Variable(1)
x_3 = Variable(1)
x_4 = Variable(1)
objective = Minimize(abs(x_1*x_2+x_3*x_4))
constraint = [x_1+x_2+x_3+x_4 == 1]
myprob = Problem(objective, constraint)

print "minimal sets:", find_minimal_sets(myprob)   # find all minimal sets
print "problem is DCP:", myprob.is_dcp()   # false
print "problem is DMCP:", is_dmcp(myprob)  # true
result = myprob.solve(method = 'dmcp')
```
The output of the above code is as follows.
```
minimal sets: [[1, 3], [1, 2], [0, 3], [0, 2]]
problem is DCP: False
problem is DMCP: True
maximum value of slack variables: 1.15081491391e-05
objective value: 1.74866042578e-05
```

The solutions obtained by DMCP depend heavily on the initial point the solving algorithm starts from.
It is strongly suggested that users set reasonable initial points.
Otherwise, the algorithm starts from a random initial point.
Users can specify an initial point manually by setting the ``value`` field of the problem variables.
For example:
```
x_1.value = 1.2
x_2.value = -3
x_3.value = 4
x_4.value = 0.15
result = myprob.solve(method = 'dmcp')
```

Multi-convex atomic functions
-----------------------------
In order to allow multi-convex functions, we extend the atomic function set of ``CVXPY``.
The following atoms are allowed to have non-constant expressions in both arguments, while in the dictionary of ``CVXPY`` the first argument must be constant.
* multiplication: ``expression1 * expression2``
* elementwise multiplication: ``mul_elemwise(expression1, expression2)``
* convolution: ``conv(expression1, expression2)``

Functions and attributes
----------------
* ``is_dmcp(problem)`` returns a boolean indicating if an optimization problem satisfies DMCP rules.
* ``find_minimal_sets(problem)`` analyzes the problem and returns a list of minimal sets of (indexes of) variables.
The indexes are with respect to the list ``problem.variables()``, namely the variable corresponding to the index ``0`` is
``problem.variables()[0]``. If the problem is DCP, it returns an empty list.
* ``fix(expression, fix_vars)`` returns a new expression with the variables in the list ``fix_vars`` replaced with parameters of the same signs and values.
* ``fix_prob(problem, fix_vars)`` returns a new problem with the given variables replaced with parameters of the same signs and values.

Constructing and solving problems
---------------------------------
The components of the variable, the objective, and the constraints are constructed using standard CVXPY syntax. Once the user has constructed a problem object, they can apply the following solve method:
* ``problem.solve(method = 'dmcp')`` applies the solving algorithm with proximal operators, and returns the number of iterations, and the maximum value of the slack variables. The solution to every variable is in its ``value`` field.
* ``problem.solve(method = 'dmcp', proximal = False)`` applies the solving method without proximal operators.
* ``problem.solve(method = 'dmcp', linearize = True)`` applies the solving method with prox-linear operators.

Additional arguments can be used to specify the parameters.

Solve method parameters:
* The ``max_iter`` parameter sets the maximum number of iterations in the algorithm. The default is 100.
* The ``mu`` parameter trades off satisfying the constraints and minimizing the objective. Larger ``mu`` favors satisfying the constraints. The default is 0.001.
* The ``rho`` parameter sets the rate at which ``mu`` increases inside the algorithm. The default is 1.2.
* The ``mu_max`` parameter upper bounds how large ``mu`` can get. The default is 1e4.
* The ``lambd`` parameter is the parameter in the proximal operator. The default is 10.
* The ``solver`` parameter specifies what solver to use to solve convex subproblems.

Any additional keyword arguments will be passed to the solver for convex subproblems. For example, ``warm_start=True`` will tell the convex solver to use a warm start.

