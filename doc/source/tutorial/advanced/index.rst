.. _advanced:

Advanced Features
=================

This section of the tutorial covers features of CVXPY intended for users with advanced knowledge of convex optimization. We recommend `Convex Optimization <https://www.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe as a reference for any terms you are unfamiliar with.

N-dimensional expressions
-------------------------

.. versionadded:: 1.6

CVXPY now supports N-dimensional expressions. This allows one to define variables, parameters, and constants with arbitrary number of dimensions.
This new feature enables users to model problems with multi-dimensional data in a more natural way.

In the example below, we consider a problem where the goal is to optimize the usage of a resource across multiple locations, days, and hours.
We are now able to easily form constraints on any combination of dimensions.

.. code:: python
    
    # create a 3-dimensional variable (locations, days, hours)
    x = cp.Variable((12, 10, 24))

    constraints = [
      cp.sum(x, axis=(0, 2)) <= 2000, # constrain the daily usage across all locations
      x[:, :, :12] <= 100 # constrain the first 12 hours of each day at every location
      x[0, 3, :] == 0 # constrain the usage at the first location on the fourth day to be zero
      ]

    obj = cp.Minimize(cp.sum_squares(x))
    prob = cp.Problem(obj, constraints)
    prob.solve()

Please refer to NumPy's excellent `reference <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ 
on N-dimensional arrays and the `array API standard <https://data-apis.org/array-api/latest/API_specification/index.html>`_ for more details
on how to manipulate N-dimensional arrays. Our goal is to match the NumPy API as closely as possible.

.. warning::

    N-dimensional support is still experimental and may not work with all CVXPY features.
    If you encounter any issues or missing functionality, please report them on `GitHub issues <https://github.com/cvxpy/cvxpy/issues>`_.

Dual variables
--------------

You can use CVXPY to find the optimal dual variables for a problem. When you call ``prob.solve()`` each dual variable in the solution is stored in the ``dual_value`` field of the constraint it corresponds to.


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
    prob.solve()

    # The optimal dual variable (Lagrange multiplier) for
    # a constraint is stored in constraint.dual_value.
    print("optimal (x + y == 1) dual variable", constraints[0].dual_value)
    print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
    print("x - y value:", (x - y).value)

::

    optimal (x + y == 1) dual variable 6.47610300459e-18
    optimal (x - y >= 1) dual variable 2.00025244976
    x - y value: 0.999999986374

The dual variable for ``x - y >= 1`` is 2. By complementarity this implies that ``x - y`` is 1, which we can see is true. The fact that the dual variable is non-zero also tells us that if we tighten ``x - y >= 1``, (i.e., increase the right-hand side), the optimal value of the problem will increase.

Transforms
----------

Transforms provide additional ways of manipulating CVXPY objects
beyond the atomic functions.  For example, the :py:class:`indicator
<cvxpy.transforms.indicator>` transform converts a list of constraints into an
expression representing the convex function that takes value 0 when the
constraints hold and :math:`\infty` when they are violated.


.. code:: python

   x = cp.Variable()
   constraints = [0 <= x, x <= 1]
   expr = cp.transforms.indicator(constraints)
   x.value = .5
   print("expr.value = ", expr.value)
   x.value = 2
   print("expr.value = ", expr.value)

::

   expr.value = 0.0
   expr.value = inf

The full set of transforms available is discussed in :ref:`transforms-api`.

Problem arithmetic
------------------

For convenience, arithmetic operations have been overloaded for
problems and objectives.
Problem arithmetic is useful because it allows you to write a problem as a
sum of smaller problems.
The rules for adding, subtracting, and multiplying objectives are given below.

.. code:: python

    # Addition and subtraction.

    Minimize(expr1) + Minimize(expr2) == Minimize(expr1 + expr2)

    Maximize(expr1) + Maximize(expr2) == Maximize(expr1 + expr2)

    Minimize(expr1) + Maximize(expr2) # Not allowed.

    Minimize(expr1) - Maximize(expr2) == Minimize(expr1 - expr2)

    # Multiplication (alpha is a positive scalar).

    alpha*Minimize(expr) == Minimize(alpha*expr)

    alpha*Maximize(expr) == Maximize(alpha*expr)

    -alpha*Minimize(expr) == Maximize(-alpha*expr)

    -alpha*Maximize(expr) == Minimize(-alpha*expr)

The rules for adding and multiplying problems are equally straightforward:

.. code:: python

    # Addition and subtraction.

    prob1 + prob2 == Problem(prob1.objective + prob2.objective,
                             prob1.constraints + prob2.constraints)

    prob1 - prob2 == Problem(prob1.objective - prob2.objective,
                             prob1.constraints + prob2.constraints)

    # Multiplication (alpha is any scalar).

    alpha*prob == Problem(alpha*prob.objective, prob.constraints)

Note that the ``+`` operator concatenates lists of constraints,
since this is the default behavior for Python lists.
The in-place operators ``+=``, ``-=``, and ``*=`` are also supported for
objectives and problems and follow the same rules as above.

.. Given the optimization problems :math:`p_1,\ldots,p_n` where each
.. :math:`p_i` is of the form

.. :math:`\begin{array}{ll}
.. \mbox{minimize}  &f_i(x) \\
.. \mbox{subject to} &x \in \mathcal C_i
.. \end{array}`

.. the weighted sum `\sum_{i=1}^n \alpha_i p_i` is the problem

.. :math:`\begin{array}{ll}
.. \mbox{minimize}  &\sum_{i=1}^n \alpha_i f_i(x) \\
.. \mbox{subject to} &x \in \cap_{i=1}^n \mathcal C_i
.. \end{array}`

Getting the standard form
-------------------------

If you are interested in getting the standard form that CVXPY produces for a
problem, you can use the ``get_problem_data`` method. When a problem is solved, 
a :class:`~cvxpy.reductions.solvers.solving_chain.SolvingChain` passes a
low-level representation that is compatible with the targeted solver to a
solver, which solves the problem. This method returns that low-level
representation, along with a ``SolvingChain`` and metadata for unpacking
a solution into the problem. This low-level representation closely resembles,
but is not identical to, the arguments supplied to the solver.

A solution to the equivalent low-level problem can be obtained via the
data by invoking the ``solve_via_data`` method of the returned solving
chain, a thin wrapper around the code external to CVXPY that further
processes and solves the problem. Invoke the ``unpack_results`` method
to recover a solution to the original problem.

For example:

.. code:: python

  problem = cp.Problem(objective, constraints)
  data, chain, inverse_data = problem.get_problem_data(cp.SCS)
  # calls SCS using `data`
  soln = chain.solve_via_data(problem, data)
  # unpacks the solution returned by SCS into `problem`
  problem.unpack_results(soln, chain, inverse_data)

Alternatively, the ``data`` dictionary returned by this method
contains enough information to bypass CVXPY and call the solver
directly.

For example:

.. code:: python

  problem = cp.Problem(objective, constraints)
  probdata, _, _ = problem.get_problem_data(cp.SCS)

  import scs
  data = {
    'A': probdata['A'],
    'b': probdata['b'],
    'c': probdata['c'],
  }
  cone_dims = probdata['dims']
  cones = {
      "f": cone_dims.zero,
      "l": cone_dims.nonpos,
      "q": cone_dims.soc,
      "ep": cone_dims.exp,
      "s": cone_dims.psd,
  }
  soln = scs.solve(data, cones)

The structure of the data dict that CVXPY returns depends on the solver. For
details, print the dictionary, or consult the solver interfaces in
``cvxpy/reductions/solvers``.

.. _canonicalization-backends:

Canonicalization backends
------------------------------------
Users can select from multiple canonicalization backends by adding the ``canon_backend``
keyword argument to the ``.solve()`` call, e.g. ``problem.solve(canon_backend=cp.SCIPY_CANON_BACKEND)``
(Introduced in CVXPY 1.3).
This can speed up the canonicalization time significantly for some problems.
Currently, the following canonicalization backends are supported:

*  CPP (default): The original C++ implementation, also referred to as CVXCORE.
*  | SCIPY: A pure Python implementation based on the SciPy sparse module.
   | Generally fast for problems that are already vectorized.
*  NUMPY: Reference implementation in pure NumPy. Fast for some small or dense problems.
