.. _intro:

What is CVXPY?
==============

CVXPY is a Python-embedded modeling language for convex optimization
problems. It
automatically transforms the problem into standard form, calls a solver,
and unpacks the results.

The code below solves a simple optimization problem in CVXPY:

.. code:: python

    import cvxpy as cvx

    # Create two scalar optimization variables.
    x = cvx.Variable()
    y = cvx.Variable()

    # Create two constraints.
    constraints = [x + y == 1,
                   x - y >= 1]

    # Form objective.
    obj = cvx.Minimize((x - y)**2)

    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value, y.value)

::

    status: optimal
    optimal value 0.999999999761
    optimal var 1.00000000001 -1.19961841702e-11


The status, which was assigned a value "optimal" by the solve method,
tells us the problem was solved successfully. The optimal value
(basically 1 here) is the minimum value of the objective over all
choices of variables that satisfy the constraints. The last thing
printed gives values of x and y (basically 1 and 0 respectively) that
achieve the optimal objective.

``prob.solve()`` returns the optimal value and updates ``prob.status``,
``prob.value``, and the ``value`` field of all the variables in the
problem.

Changing the problem
--------------------

:py:class:`Problems <cvxpy.problems.problem.Problem>` are immutable, meaning they
cannot be changed after they are created.  To change the objective or
constraints, create a new problem.

.. code:: python

    # Replace the objective.
    prob2 = cvx.Problem(cvx.Maximize(x + y), prob.constraints)
    print("optimal value", prob2.solve())

    # Replace the constraint (x + y == 1).
    constraints = [x + y <= 3] + prob.constraints[1:]
    prob2 = cvx.Problem(prob.objective, constraints)
    print("optimal value", prob2.solve())

::

    optimal value 1.0
    optimal value 3.00000000006


Infeasible and unbounded problems
---------------------------------

If a problem is infeasible or unbounded, the status field will be set to
"infeasible" or "unbounded", respectively. The value fields of the
problem variables are not updated.

.. code:: python

    import cvxpy as cvx

    x = cvx.Variable()

    # An infeasible problem.
    prob = cvx.Problem(cvx.Minimize(x), [x >= 1, x <= 0])
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)

    # An unbounded problem.
    prob = cvx.Problem(cvx.Minimize(x))
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)

::

    status: infeasible
    optimal value inf
    status: unbounded
    optimal value -inf


Notice that for a minimization problem the optimal value is ``inf`` if
infeasible and ``-inf`` if unbounded. For maximization problems the
opposite is true.

Other problem statuses
----------------------

If the solver called by CVXPY solves the problem but to a lower accuracy than desired, the
problem status indicates the lower accuracy achieved. The
statuses indicating lower accuracy are

* "optimal\_inaccurate"
* "unbounded\_inaccurate"
* "infeasible\_inaccurate"

The problem variables are updated as usual for the type of solution
found (i.e., optimal, unbounded, or infeasible).

If the solver completely fails to solve the problem, CVXPY throws a ``SolverError`` exception.
If this happens you should try using other solvers. See
the discussion of :ref:`solvers` for details.

CVXPY provides the following constants as aliases for the different status strings:

* ``OPTIMAL``
* ``INFEASIBLE``
* ``UNBOUNDED``
*  ``OPTIMAL_INACCURATE``
* ``INFEASIBLE_INACCURATE``
* ``UNBOUNDED_INACCURATE``

For example, to test if a problem was solved successfully, you would use

.. code:: python

    prob.status == OPTIMAL

Vectors and matrices
--------------------

:py:class:`Variables <cvxpy.expressions.variable.Variable>` can be scalars,
vectors, or matrices, meaning they are 0, 1, or 2 dimensional.


.. code:: python

    # A scalar variable.
    a = cvx.Variable()

    # Vector variable with shape (5,).
    x = cvx.Variable(5)

    # Matrix variable with shape (5, 1).
    x = cvx.Variable((5, 1))

    # Matrix variable with shape (4, 7).
    A = cvx.Variable((4, 7))

You can use your numeric library of choice to construct matrix and
vector constants. For instance, if ``x`` is a CVXPY Variable in the
expression ``A*x + b``, ``A`` and ``b`` could be Numpy ndarrays, SciPy
sparse matrices, etc. ``A`` and ``b`` could even be different types.

Currently the following types may be used as constants:

-  NumPy ndarrays
-  NumPy matrices
-  SciPy sparse matrices

Here's an example of a CVXPY problem with vectors and matrices:

.. code:: python

    # Solves a bounded least-squares problem.

    import cvxpy as cvx
    import numpy

    # Problem data.
    m = 10
    n = 5
    numpy.random.seed(1)
    A = numpy.random.randn(m, n)
    b = numpy.random.randn(m)

    # Construct the problem.
    x = cvx.Variable(n)
    objective = cvx.Minimize(cvx.sum_squares(A*x - b))
    constraints = [0 <= x, x <= 1]
    prob = cvx.Problem(objective, constraints)

    print("Optimal value", prob.solve())
    print("Optimal var")
    print(x.value) # A numpy ndarray.

::

    Optimal value 4.14133859146
    Optimal var
    [ -5.11480673e-21   6.30625742e-21   1.34643668e-01   1.24976681e-01
  -4.79039542e-21]

Constraints
-----------

As shown in the example code, you can use ``==``, ``<=``, and ``>=`` to construct constraints in CVXPY. Equality and inequality constraints are elementwise, whether they involve scalars, vectors, or matrices. For example, together the constraints ``0 <= x`` and ``x <= 1`` mean that every entry of ``x`` is between 0 and 1.

If you want matrix inequalities that represent semi-definite cone constraints, see :ref:`semidefinite`. The section explains how to express a semi-definite cone inequality.

You cannot construct inequalities with ``<`` and ``>``. Strict inequalities don't make sense in a real world setting. Also, you cannot chain constraints together, e.g., ``0 <= x <= 1`` or ``x == y == 2``. The Python interpreter treats chained constraints in such a way that CVXPY cannot capture them. CVXPY will raise an exception if you write a chained constraint.

Parameters
----------

:py:class:`Parameters <cvxpy.expressions.constants.parameter.Parameter>` are symbolic
representations of constants. The purpose of parameters is to change the value
of a constant in a problem without reconstructing the entire
problem.

Parameters can be vectors or matrices, just like variables. When you
create a parameter you have the option of specifying attributes such as the
sign of the parameter's entries, whether the parameter is symmetric, etc.
These attributes are used in :ref:`dcp` and are unknown unless specified.
Parameters can be assigned a constant value any time after they are created.
The constant value must have the same dimensions and attributes
as those specified when the parameter was created.

.. code:: python

    # Positive scalar parameter.
    m = cvx.Parameter(nonneg=True)

    # Column vector parameter with unknown sign (by default).
    c = cvx.Parameter(5)

    # Matrix parameter with negative entries.
    G = cvx.Parameter((4, 7), nonpos=True)

    # Assigns a constant value to G.
    G.value = -numpy.ones((4, 7))

You can initialize a parameter with a value. The following code segments are equivalent:

.. code:: python

    # Create parameter, then assign value.
    rho = cvx.Parameter(nonneg=True)
    rho.value = 2

    # Initialize parameter with a value.
    rho = cvx.Parameter(nonneg=True, value=2)

Computing trade-off curves is a common use of parameters. The example below
computes a trade-off curve for a LASSO problem.

.. code:: python

    import cvxpy as cvx
    import numpy
    import matplotlib.pyplot as plt

    # Problem data.
    n = 15
    m = 10
    numpy.random.seed(1)
    A = numpy.random.randn(n, m)
    b = numpy.random.randn(n)
    # gamma must be nonnegative due to DCP rules.
    gamma = cvx.Parameter(nonneg=True)

    # Construct the problem.
    x = cvx.Variable(m)
    error = cvx.sum_squares(A*x - b)
    obj = cvx.Minimize(error + gamma*cvx.norm(x, 1))
    prob = cvx.Problem(obj)

    # Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
    sq_penalty = []
    l1_penalty = []
    x_values = []
    gamma_vals = numpy.logspace(-4, 6)
    for val in gamma_vals:
        gamma.value = val
        prob.solve()
        # Use expr.value to get the numerical value of
        # an expression in the problem.
        sq_penalty.append(error.value)
        l1_penalty.append(cvx.norm(x, 1).value)
        x_values.append(x.value)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(6,10))

    # Plot trade-off curve.
    plt.subplot(211)
    plt.plot(l1_penalty, sq_penalty)
    plt.xlabel(r'\|x\|_1', fontsize=16)
    plt.ylabel(r'\|Ax-b\|^2', fontsize=16)
    plt.title('Trade-Off Curve for LASSO', fontsize=16)

    # Plot entries of x vs. gamma.
    plt.subplot(212)
    for i in range(m):
        plt.plot(gamma_vals, [xi[i] for xi in x_values])
    plt.xlabel(r'\gamma', fontsize=16)
    plt.ylabel(r'x_{i}', fontsize=16)
    plt.xscale('log')
    plt.title(r'\text{Entries of x vs. }\gamma', fontsize=16)

    plt.tight_layout()
    plt.show()


.. image:: tutorial_files/tutorial_20_0.png


Trade-off curves can easily be computed in parallel. The code below
computes in parallel the optimal x for each :math:`\gamma` in the LASSO
problem above.

.. code:: python

    from multiprocessing import Pool

    # Assign a value to gamma and find the optimal x.
    def get_x(gamma_value):
        gamma.value = gamma_value
        result = prob.solve()
        return x.value

    # Parallel computation (set to 1 process here).
    pool = Pool(processes = 1)
    x_values = pool.map(get_x, gamma_vals)
