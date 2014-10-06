.. _advanced:

Advanced Features
=================

This section of the tutorial covers features of CVXPY intended for users with advanced knowledge of convex optimization. We recommend `Convex Optimization <http://www.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe as a reference for any terms you are unfamiliar with.

Dual variables
--------------

You can use CVXPY to find the optimal dual variables for a problem. When you call ``prob.solve()`` each dual variable in the solution is stored in the ``dual_value`` field of the constraint it corresponds to.


.. code:: python

    from cvxpy import *

    # Create two scalar optimization variables.
    x = Variable()
    y = Variable()

    # Create two constraints.
    constraints = [x + y == 1,
                   x - y >= 1]

    # Form objective.
    obj = Minimize(square(x - y))

    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()

    # The optimal dual variable (Lagrange multiplier) for
    # a constraint is stored in constraint.dual_value.
    print "optimal (x + y == 1) dual variable", constraints[0].dual_value
    print "optimal (x - y >= 1) dual variable", constraints[1].dual_value
    print "x - y value:", (x - y).value

.. parsed-literal::

    optimal (x + y == 1) dual variable 6.47610300459e-18
    optimal (x - y >= 1) dual variable 2.00025244976
    x - y value: 0.999999986374

The dual variable for ``x - y >= 1`` is 2. By complementarity this implies that ``x - y`` is 1, which we can see is true. The fact that the dual variable is non-zero also tells us that if we tighten ``x - y >= 1``, (i.e., increase the right-hand side), the optimal value of the problem will increase.


.. _semidefinite:

Semidefinite matrices
----------------------

Many convex optimization problems involve constraining matrices to be positive or negative semidefinite (e.g., SDPs). You can do this in CVXPY using the ``Semidef`` constructor. ``Semidef(n)`` constructs an ``n`` by ``n`` variable constrained to be positive semidefinite. For example,

.. code:: python

    # Creates a 100 by 100 positive semidefinite variable.
    X = Semidef(100)

    # You can use X anywhere you would use
    # a normal CVXPY variable.
    obj = Minimize(norm(X) + sum_entries(X))

The following code shows how to use ``Semidef`` to constrain matrix expressions to be positive or negative semidefinite:

.. code:: python

    # expr1 must be positive semidefinite.
    constr1 = (expr1 == Semidef(n))

    # expr2 must be negative semidefinite.
    constr2 = (expr2 == -Semidef(n))

To constrain a matrix expression to be symmetric (but not necessarily positive or negative semidefinite), simply write

.. code:: python

    # expr must be symmetric.
    constr = (expr == expr.T)

Solve method options
--------------------

The ``solve`` method takes optional arguments that let you change how CVXPY solves the problem. Here is the signature for the ``solve`` method:

.. function:: solve(solver=None, verbose=False, **kwargs)

   Solves a DCP compliant optimization problem.

   :param solver: The solver to use.
   :type solver: str, optional
   :param verbose:  Overrides the default of hiding solver output.
   :type verbose: bool, optional
   :param kwargs: Additional keyword arguments specifying solver specific options.
   :return: The optimal value for the problem, or a string indicating why the problem could not be solved.

We will discuss the optional arguments in detail below.

.. _solvers:

Choosing a solver
^^^^^^^^^^^^^^^^^

CVXPY uses the open source solvers `ECOS`_, `CVXOPT`_, and `SCS`_. The table below shows the types of problems the solvers can handle.

+-----------+----+------+-----+-----+
|           | LP | SOCP | SDP | EXP |
+===========+====+======+=====+=====+
| `ECOS`_   | X  | X    |     |     |
+-----------+----+------+-----+-----+
| `CVXOPT`_ | X  | X    | X   | X   |
+-----------+----+------+-----+-----+
| `SCS`_    | X  | X    | X   | X   |
+-----------+----+------+-----+-----+

Here EXP refers to problems with exponential cone constraints. The exponential cone is defined as

    :math:`\{(x,y,z) \mid y > 0, y\exp(x/y) \leq z \} \cup \{ (x,y,z) \mid x \leq 0, y = 0, z \geq 0\}`.

You cannot specify cone constraints explicitly in CVXPY, but cone constraints are added when CVXPY converts the problem into standard form.

By default CVXPY calls the solver most specialized to the problem type. For example, `ECOS`_ is called for SOCPs. `SCS`_ and `CVXOPT`_ can both handle all problems. `CVXOPT`_ is preferred by default. For many problems `SCS`_ will be faster, though less accurate.

You can change the solver called by CVXPY using the ``solver`` keyword argument. If the solver you choose cannot solve the problem, CVXPY will raise an exception. Here's example code solving the same problem with different solvers.

.. code:: python

    # Solving a problem with different solvers.
    x = Variable(2)
    obj = Minimize(norm(x, 2) + norm(x, 1))
    constraints = [x >= 2]
    prob = Problem(obj, constraints)

    # Solve with ECOS.
    prob.solve(solver=ECOS)
    print "optimal value with ECOS:", prob.value

    # Solve with CVXOPT.
    prob.solve(solver=CVXOPT)
    print "optimal value with CVXOPT:", prob.value

    # Solve with SCS.
    prob.solve(solver=SCS)
    print "optimal value with SCS:", prob.value

.. parsed-literal::

    optimal value with ECOS: 6.82842708233
    optimal value with CVXOPT: 6.82842708994
    optimal value with SCS: 6.82837896978

Viewing solver output
^^^^^^^^^^^^^^^^^^^^^

All the solvers can print out information about their progress while solving the problem. This information can be useful in debugging a solver error. To see the output from the solvers, set ``verbose=True`` in the solve method.

.. code:: python

    # Solve with ECOS and display output.
    prob.solve(solver=ECOS, verbose=True)
    print "optimal value with ECOS:", prob.value

.. parsed-literal::

    ECOS 1.0.3 - (c) A. Domahidi, Automatic Control Laboratory, ETH Zurich, 2012-2014.

    It     pcost         dcost      gap     pres    dres     k/t     mu      step     IR
     0   +0.000e+00   +4.000e+00   +2e+01   2e+00   1e+00   1e+00   3e+00    N/A     1 1 -
     1   +6.451e+00   +8.125e+00   +5e+00   7e-01   5e-01   7e-01   7e-01   0.7857   1 1 1
     2   +6.788e+00   +6.839e+00   +9e-02   1e-02   8e-03   3e-02   2e-02   0.9829   1 1 1
     3   +6.828e+00   +6.829e+00   +1e-03   1e-04   8e-05   3e-04   2e-04   0.9899   1 1 1
     4   +6.828e+00   +6.828e+00   +1e-05   1e-06   8e-07   3e-06   2e-06   0.9899   2 1 1
     5   +6.828e+00   +6.828e+00   +1e-07   1e-08   8e-09   4e-08   2e-08   0.9899   2 1 1

    OPTIMAL (within feastol=1.3e-08, reltol=1.5e-08, abstol=1.0e-07).
    Runtime: 0.000121 seconds.

    optimal value with ECOS: 6.82842708233

Setting solver options
^^^^^^^^^^^^^^^^^^^^^^

The `ECOS`_, `CVXOPT`_, and `SCS`_ Python interfaces allow you to set solver options such as the maximum number of iterations. You can pass these options along through CVXPY as keyword arguments.

For example, here we tell SCS to use an indirect method for solving linear equations rather than a direct method.

.. code:: python

    # Solve with SCS, use sparse-indirect method.
    prob.solve(solver=SCS, verbose=True, use_indirect=True)
    print "optimal value with SCS:", prob.value

.. parsed-literal::

    ----------------------------------------------------------------------------
        SCS v1.0.5 - Splitting Conic Solver
        (c) Brendan O'Donoghue, Stanford University, 2012
    ----------------------------------------------------------------------------
    Lin-sys: sparse-indirect, nnz in A = 13, CG tol ~ 1/iter^(2.00)
    EPS = 1.00e-03, ALPHA = 1.80, MAX_ITERS = 2500, NORMALIZE = 1, SCALE = 5.00
    Variables n = 5, constraints m = 9
    Cones:  linear vars: 6
        soc vars: 3, soc blks: 1
    Setup time: 2.78e-04s
    ----------------------------------------------------------------------------
     Iter | pri res | dua res | rel gap | pri obj | dua obj | kap/tau | time (s)
    ----------------------------------------------------------------------------
         0| 4.60e+00  5.78e-01       nan      -inf       inf       inf  3.86e-05
        60| 3.92e-05  1.12e-04  6.64e-06  6.83e+00  6.83e+00  1.41e-17  9.51e-05
    ----------------------------------------------------------------------------
    Status: Solved
    Timing: Total solve time: 9.76e-05s
        Lin-sys: avg # CG iterations: 1.00, avg solve time: 2.24e-07s
        Cones: avg projection time: 4.90e-08s
    ----------------------------------------------------------------------------
    Error metrics:
    |Ax + s - b|_2 / (1 + |b|_2) = 3.9223e-05
    |A'y + c|_2 / (1 + |c|_2) = 1.1168e-04
    |c'x + b'y| / (1 + |c'x| + |b'y|) = 6.6446e-06
    dist(s, K) = 0, dist(y, K*) = 0, s'y = 0
    ----------------------------------------------------------------------------
    c'x = 6.8284, -b'y = 6.8285
    ============================================================================
    optimal value with SCS: 6.82837896975

Here's the complete list of solver options.

`ECOS`_ options:

``'max_iters'``
    maximum number of iterations (default: 100).

``'abstol'``
    absolute accuracy (default: 1e-7).

``'reltol'``
    relative accuracy (default: 1e-6).

``'feastol'``
    tolerance for feasibility conditions (default: 1e-7).

``'abstol_inacc'``
    absolute accuracy for inaccurate solution (default: 5e-5).

``'reltol_inacc'``
    relative accuracy for inaccurate solution (default: 5e-5).

``'feastol_inacc'``
    tolerance for feasibility condition for inaccurate solution (default: 1e-4).

`CVXOPT`_ options:

``'max_iters'``
    maximum number of iterations (default: 100).

``'abstol'``
    absolute accuracy (default: 1e-7).

``'reltol'``
    relative accuracy (default: 1e-6).

``'feastol'``
    tolerance for feasibility conditions (default: 1e-7).

`SCS`_ options:

``'max_iters'``
    maximum number of iterations (default: 2500).

``'eps'``
    convergence tolerance (default: 1e-3).

``'alpha'``
    relaxation parameter (default: 1.8).

``'normalize'``
    whether to precondition data matrices (default: True).

``'use_indirect'``
    whether to use indirect solver for KKT sytem (instead of direct) (default: False).

Getting the standard form
-------------------------

If you are interested in getting the standard form that CVXPY produces for a problem, you can use the ``get_problem_data`` method. Calling ``get_problem_data(solver)`` on a problem object returns the arguments that CVXPY would pass to that solver. If the solver you choose cannot solve the problem, CVXPY will raise an exception.

.. code:: python

    # Get ECOS arguments.
    c, G, h, dims, A, b = prob.get_problem_data(ECOS)

    # Get CVXOPT arguments.
    c, G, h, dims, A, b = prob.get_problem_data(CVXOPT)

    # Get SCS arguments.
    data, dims = prob.get_problem_data(SCS)

After you solve the standard conic form problem returned by ``get_problem_data``, you can unpack the raw solver output using the ``unpack_results`` method. Calling ``unpack_results(solver, solver_output)`` on a problem will update the values of all primal and dual variables as well as the problem value and status.

For example, the following code is equivalent to solving the problem directly with CVXPY:

.. code:: python

    # Get ECOS arguments.
    c, G, h, dims, A, b = prob.get_problem_data(ECOS)
    # Call ECOS solver.
    solver_output = ecos.solve(c, G, h, dims, A, b)
    # Unpack raw solver output.
    prob.unpack_results(ECOS, solver_output)

.. _CVXOPT: http://cvxopt.org/
.. _ECOS: http://github.com/ifa-ethz/ecos
.. _SCS: http://github.com/cvxgrp/scs