.. _intro:

What is CVXPY?
==============

CVXPY is a Python-embedded modeling language for convex optimization
problems. It
automatically transforms the problem into standard form, calls a solver,
and unpacks the results.

The code below solves a simple optimization problem in CVXPY:

.. code:: python

    import cvxpy as cp

    # Create two scalar optimization variables.
    x = cp.Variable()
    y = cp.Variable()

    # Create two constraints.
    constraints = [x + y == 1,
                   x - y >= 1]

    # Form objective.
    obj = cp.Minimize((x - y)**2)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
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
    prob2 = cp.Problem(cp.Maximize(x + y), prob.constraints)
    print("optimal value", prob2.solve())

    # Replace the constraint (x + y == 1).
    constraints = [x + y <= 3] + prob2.constraints[1:]
    prob3 = cp.Problem(prob2.objective, constraints)
    print("optimal value", prob3.solve())

::

    optimal value 1.0
    optimal value 3.00000000006


Infeasible and unbounded problems
---------------------------------

If a problem is infeasible or unbounded, the status field will be set to
"infeasible" or "unbounded", respectively. The value fields of the
problem variables are not updated.

.. code:: python

    import cvxpy as cp

    x = cp.Variable()

    # An infeasible problem.
    prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)

    # An unbounded problem.
    prob = cp.Problem(cp.Minimize(x))
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
* ``INFEASIBLE_OR_UNBOUNDED``

To test if a problem was solved successfully, you would use

.. code:: python

    prob.status == OPTIMAL

The status ``INFEASIBLE_OR_UNBOUNDED`` is rare. It's used when a solver was able to
determine that the problem was either infeasible or unbounded, but could not tell which.
You can determine the precise status by re-solving the problem where you
set the objective function to a constant (e.g., ``objective = cp.Minimize(0)``).
If the new problem is solved with status code ``INFEASIBLE_OR_UNBOUNDED`` then the
original problem was infeasible. If the new problem is solved with status ``OPTIMAL``
then the original problem was unbounded.

Vectors and matrices
--------------------

:py:class:`Variables <cvxpy.expressions.variable.Variable>` can be scalars,
vectors, or matrices, meaning they are 0, 1, or 2 dimensional.


.. code:: python

    # A scalar variable.
    a = cp.Variable()

    # Vector variable with shape (5,).
    x = cp.Variable(5)

    # Column vector variable with shape (5, 1).
    x = cp.Variable((5, 1))

    # Matrix variable with shape (4, 7).
    A = cp.Variable((4, 7))

You can use your numeric library of choice to construct matrix and
vector constants. For instance, if ``x`` is a CVXPY Variable in the
expression ``A @ x + b``, ``A`` and ``b`` could be Numpy ndarrays, SciPy
sparse matrices, etc. ``A`` and ``b`` could even be different types.

Currently the following types may be used as constants:

-  NumPy ndarrays
-  SciPy sparse matrices

Here's an example of a CVXPY problem with vectors and matrices:

.. code:: python

    # Solves a bounded least-squares problem.
    import cvxpy as cp
    import numpy as np

    # Problem data.
    m = 10
    n = 5
    numpy.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # Construct the problem.
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)

    print("Optimal objective value", prob.solve())
    print("Optimal variable value")
    print(x.value) # A numpy ndarray.

::

    Optimal objective value 4.14133859146
    Optimal variable value
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
of a constant in a problem without reconstructing the entire problem. In many
cases, solving a parametrized program multiple times can be
substantially faster than repeatedly solving a new problem: after reading
this section, be sure to read the tutorial on :ref:`dpp` (DPP).

When you create a parameter you have the option of specifying attributes such as the
sign of the parameter's entries, whether the parameter is symmetric, etc.
These attributes are used in :ref:`dcp` and are unknown unless specified.
Parameters can be assigned a constant value any time after they are created.
The constant value must have the same dimensions and attributes
as those specified when the parameter was created.

.. code:: python

    # Positive scalar parameter.
    m = cp.Parameter(nonneg=True)

    # Column vector parameter with unknown sign (by default).
    c = cp.Parameter(5)

    # Matrix parameter with negative entries.
    G = cp.Parameter((4, 7), nonpos=True)

    # Assigns a constant value to G.
    G.value = -np.ones((4, 7))

You can initialize a parameter with a value. The following code segments are equivalent:

.. code:: python

    # Create parameter, then assign value.
    rho = cp.Parameter(nonneg=True)
    rho.value = 2

    # Initialize parameter with a value.
    rho = cp.Parameter(nonneg=True, value=2)

Computing trade-off curves is a common use of parameters. The example below
computes a trade-off curve for a LASSO problem.

.. code:: python

    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt

    # Problem data.
    n = 15
    m = 10
    np.random.seed(1)
    A = np.random.randn(n, m)
    b = np.random.randn(n)
    # gamma must be nonnegative due to DCP rules.
    gamma = cp.Parameter(nonneg=True)

    # Construct the problem.
    x = cp.Variable(m)
    error = cp.sum_squares(A @ x - b)
    obj = cp.Minimize(error + gamma*cp.norm(x, 1))
    prob = cp.Problem(obj)

    # Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
    sq_penalty = []
    l1_penalty = []
    x_values = []
    gamma_vals = np.logspace(-4, 6)
    for val in gamma_vals:
        gamma.value = val
        prob.solve()
        # Use expr.value to get the numerical value of
        # an expression in the problem.
        sq_penalty.append(error.value)
        l1_penalty.append(cp.norm(x, 1).value)
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

.. _labels:

Custom Labels
-------------

You can assign custom labels to expressions and constraints to make 
debugging and model interpretation easier. Labels appear when printing 
constraints and can be used with the ``format_labeled()`` method to 
show labeled expressions in problems.

Labels can be assigned using the ``set_label()`` method or the ``label`` property:

.. code:: python

    import cvxpy as cp
    import numpy as np
    
    # Create variables
    weights = cp.Variable(3, name="weights")
    
    # Create constraints with custom labels
    constraints = [
        (weights >= 0).set_label("non_negative_weights"),
        (cp.sum(weights) == 1).set_label("budget_constraint"),
        (weights <= 0.4).set_label("concentration_limits")
    ]
    
    # Create expressions with custom labels  
    data = np.random.randn(3)
    data_fit = cp.sum_squares(weights - data).set_label("data_fit")
    l2_reg = cp.norm(weights, 2).set_label("l2_regularization")
    
    # Build objective
    objective = cp.Minimize(data_fit + 0.5 * l2_reg)
    
    # Create and display the problem
    problem = cp.Problem(objective, constraints)
    
    # Use format_labeled() to see labels in the objective
    print(problem.format_labeled())

::

    minimize data_fit + 0.5 * l2_regularization
    subject to non_negative_weights: 0.0 <= weights
               budget_constraint: Sum(weights, None, False) == 1.0
               concentration_limits: weights <= 0.4

The ``set_label()`` method returns the object itself, allowing method chaining.
Labels are "live" and can be modified after problem creation:

.. code:: python

    # Change or remove labels dynamically
    l2_reg.label = "ridge_penalty"  # Change label
    data_fit.label = None  # Remove label
    print(problem.format_labeled())

::

    minimize sum_squares(weights + ...) + 0.5 * ridge_penalty
    subject to non_negative_weights: 0.0 <= weights
               budget_constraint: Sum(weights, None, False) == 1.0
               concentration_limits: weights <= 0.4

For more details on the label feature, including advanced usage and limitations, 
see the full labels documentation.

Next steps
----------

For more features and examples, explore the rest of the CVXPY documentation.
