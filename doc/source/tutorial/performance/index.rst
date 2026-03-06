.. _performance:

Performance Tips
================

This page provides guidance on how to write CVXPY code that compiles and solves efficiently.
CVXPY problems have two main cost components: **compile time** (how long CVXPY takes to transform
your problem for a solver) and **solve time** (how long the solver takes to find a solution).
The tips below primarily address compile time, which is often the bottleneck for large problems.

.. _vectorization:

Vectorize your problem
----------------------

The single most impactful thing you can do to reduce compile time is to **vectorize** your CVXPY
expressions. This means expressing constraints and objectives over entire vectors or matrices at
once, rather than writing scalar operations in Python loops.

As a rule of thumb, you should minimize the number (and not just the dimension) of CVXPY ``Variable``, ``Constraint``, and
``Expression`` objects needed to specify your model. Each CVXPY object adds overhead during
compilation, so fewer objects means faster compile times.

**Bad (scalarized) — slow:**

.. code:: python

    import cvxpy as cp
    import numpy as np

    m, n = 500, 200
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    x = cp.Variable(n)

    # Slow: creates m separate Constraint objects
    constraints = [A[i, :] @ x == b[i] for i in range(m)]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), constraints)
    prob.solve()

**Good (vectorized) — fast:**

.. code:: python

    import cvxpy as cp
    import numpy as np

    m, n = 500, 200
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    x = cp.Variable(n)

    # Fast: creates a single Constraint object
    constraints = [A @ x == b]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), constraints)
    prob.solve()

The vectorized version creates a single constraint object instead of ``m`` separate ones.
For large ``m``, this difference can be the gap between milliseconds and seconds of compile time.

Similarly, for simple bound constraints, consider using the ``bounds`` attribute on the variable directly, which is the most efficient approach:

.. code:: python

    # Slow
    constraints = [x[i] >= 0 for i in range(n)] + [x[i] <= 1 for i in range(n)]

    # Better (vectorized)
    constraints = [x >= 0, x <= 1]

    # Best (use bounds attribute)
    x = cp.Variable(n, bounds=[0, 1])

The ``bounds`` attribute is the most efficient way to specify simple variable bounds.
It supports scalars, NumPy arrays, parameters, and affine functions of parameters:

.. code:: python

    import numpy as np
    import cvxpy as cp

    n = 10
    # Scalar bounds
    x = cp.Variable(n, bounds=[0, 1])

    # NumPy array bounds
    lb = np.zeros(n)
    ub = np.ones(n)
    x = cp.Variable(n, bounds=[lb, ub])

    # Parameter bounds (useful for repeated solves with changing bounds)
    lb_param = cp.Parameter(n, nonneg=True)
    ub_param = cp.Parameter(n, nonneg=True)
    x = cp.Variable(n, bounds=[lb_param, ub_param])

.. _use-cvxpy-sum:

Use cp.sum, not Python's built-in sum
--------------------------------------

When summing CVXPY expressions, always use ``cp.sum`` rather than Python's built-in ``sum``.
Python's ``sum`` builds up a chain of binary ``+`` operations, creating a deep expression tree
that is slow to compile. ``cp.sum`` handles the entire sum in a single, efficient operation.

.. code:: python

    exprs = [cp.square(x[i]) for i in range(n)]

    # Slow: creates a deep binary tree of additions
    objective = cp.Minimize(sum(exprs))

    # Fast: single efficient operation
    objective = cp.Minimize(cp.sum(cp.square(x)))

.. _use-parameters:

Use parameters for repeated solves
------------------------------------

If you need to solve the same problem multiple times with different data values, use
:class:`~cvxpy.atoms.affine.add_expr.Parameter` objects instead of creating a new problem
each time. Parameters, used correctly, allow CVXPY to compile the problem structure once and reuse it across
solves, which is known as **DPP (Disciplined Parameterized Programming)**.

.. code:: python

    import cvxpy as cp
    import numpy as np

    n = 100
    x = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    data = cp.Parameter((n,))

    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - data) + gamma * cp.norm1(x)))

    # First solve — compiles and caches the problem structure
    gamma.value = 0.1
    data.value = np.random.randn(n)
    prob.solve()

    # Subsequent solves — reuses compiled structure, much faster
    gamma.value = 1.0
    data.value = np.random.randn(n)
    prob.solve()

You can verify your problem is DPP-compliant by calling ``prob.is_dpp()``.

.. _canon-backends:

Choose the right canonicalization backend
------------------------------------------

CVXPY supports multiple canonicalization backends that can significantly affect compile time
depending on your problem structure. You can select a backend via the ``canon_backend``
keyword argument to ``.solve()``:

.. code:: python

    prob.solve(canon_backend=cp.SCIPY_CANON_BACKEND)

The available backends are:

- **CPP** (default): The original C++ implementation. Works well for most problems.
- **SCIPY**: A pure Python implementation using SciPy sparse matrices. Often faster for
  already-vectorized problems.
- **COO**: A pure Python implementation using 3D COO sparse tensors. Best for
  DPP-compliant problems with large parameters.

.. _verbose-diagnostics:

Use verbose mode to diagnose slow problems
-------------------------------------------

Solving with ``verbose=True`` prints useful diagnostic information:

.. code:: python

    prob.solve(verbose=True)

The output includes DCP verification time, expression tree node count, and a
time breakdown for each compilation step. A large node count strongly suggests
the problem needs vectorization.

.. _performance-summary:

Summary of tips
---------------

+----------------------------------------------+--------------------------------------------------+
| Tip                                          | Impact                                           |
+==============================================+==================================================+
| Vectorize constraints and objectives         | Very high — can reduce compile time by orders    |
|                                              | of magnitude                                     |
+----------------------------------------------+--------------------------------------------------+
| Use ``cp.sum`` instead of Python ``sum``     | High for large sums                              |
+----------------------------------------------+--------------------------------------------------+
| Use parameters for repeated solves (DPP)     | High — amortizes compile cost across solves      |
+----------------------------------------------+--------------------------------------------------+
| Choose the right canonicalization backend    | Moderate — problem-dependent                     |
+----------------------------------------------+--------------------------------------------------+
| Use ``verbose=True`` to find bottlenecks     | Diagnostic — helps identify what to fix          |
+----------------------------------------------+--------------------------------------------------+

For a detailed benchmark, see the
`original notebook <https://github.com/cvxpy/cvxpy/blob/1.0/examples/notebooks/building_models_with_fast_compile_times.ipynb>`_
that inspired this page.
