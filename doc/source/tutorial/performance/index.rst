.. _performance:

Performance Tips
================

This page provides guidance on how to write CVXPY code that compiles and solves efficiently.
CVXPY problems have two main cost components: **compile time** (how long CVXPY takes to transform
your problem for a solver) and **solve time** (how long the solver takes to find a solution).
The tips below primarily address compile time, which can be the bottleneck for large problems.

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
    constraints = []
    for i in range(m):
        constraints.append(A[i, :] @ x == b[i])
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
    lb_param = cp.Parameter(n)
    ub_param = cp.Parameter(n)
    x = cp.Variable(n, bounds=[lb_param, ub_param])

.. _use-cvxpy-sum:

Use cp.sum, not Python's built-in sum
--------------------------------------

When summing CVXPY expressions, always use ``cp.sum`` rather than Python's built-in ``sum``.
Python's ``sum`` builds up a chain of binary ``+`` operations, creating a deep expression tree
that is slow to compile. ``cp.sum`` handles the entire sum in a single, efficient operation.

.. code:: python

    # Slow: Python's sum() creates a deep binary tree of additions
    objective = cp.Minimize(sum(cp.square(x)))

    # Fast: single efficient operation
    objective = cp.Minimize(cp.sum(cp.square(x)))

.. _use-parameters:

Use parameters for repeated solves
------------------------------------

If you need to solve the same problem multiple times with different data values, use
:class:`~cvxpy.expressions.constants.parameter.Parameter` objects instead of creating a new problem
each time. Parameters, used correctly, allow CVXPY to compile the problem structure once and reuse it across
solves, which is known as **DPP (Disciplined Parameterized Programming)**.

.. code:: python

    import cvxpy as cp
    import numpy as np

    n = 100
    x = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    data = cp.Parameter(n)

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

.. _dpp-breaking-patterns:

Patterns that silently break DPP caching
------------------------------------------

A problem can be DCP-valid (it compiles and solves) yet not DPP-compliant,
which means CVXPY re-compiles it from scratch on every solve. Because there
is no warning by default, this is a common source of unexpectedly slow
repeated solves. Below are the three most common patterns and their fixes.

**1. Parametric quadratic forms in constraints**

``cp.quad_form(x, P)`` is DPP-compliant when ``P`` is a Parameter **in the
objective** of a solver that supports quadratic objectives (e.g., Clarabel,
OSQP). However, the same expression in a **constraint** is *not*
DPP-compliant — CVXPY silently falls back to full recompilation:

.. code:: python

    import cvxpy as cp
    import numpy as np

    n = 10
    x = cp.Variable(n)
    P = cp.Parameter((n, n), PSD=True)

    # DPP in the objective — fast re-solves
    obj_prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)))
    print(obj_prob.is_dpp())  # True

    # NOT DPP in a constraint — silent slow path
    con_prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [cp.quad_form(x, P) <= 1])
    print(con_prob.is_dpp())  # False

When you need a parametric quadratic form in a constraint, use a Cholesky
factorization: pass ``L = np.linalg.cholesky(P_value)`` as a Parameter and
write ``cp.sum_squares(L_param @ x)`` instead. See the
:ref:`DPP tutorial <dpp>` for a worked example.

**2. Unsigned parameters multiplied by concave atoms**

CVXPY's DCP rules need to know the **sign** of a parameter when it multiplies
a non-affine expression. A ``cp.Parameter`` with no sign attribute has unknown
sign, which can cause the expression to fail DCP entirely — not just DPP:

.. code:: python

    import cvxpy as cp

    n = 5
    y = cp.Variable(n, pos=True)

    # Fails DCP — sign of b is unknown, so b @ cp.log(y) is ambiguous
    b_bad = cp.Parameter(n)
    print(cp.Problem(cp.Maximize(b_bad @ cp.log(y))).is_dcp())  # False

    # Works — nonneg=True resolves the sign ambiguity
    b_good = cp.Parameter(n, nonneg=True)
    print(cp.Problem(cp.Maximize(b_good @ cp.log(y))).is_dcp())  # True

This comes up in risk-budgeting problems, entropy maximization, and any
formulation with a logarithmic or exponential barrier.

**3. Products of two parametrized expressions**

DPP requires each product to have at most one parametrized factor. A product
of two parameters is quadratic in the parameters and breaks DPP:

.. code:: python

    import cvxpy as cp

    x = cp.Variable()
    a = cp.Parameter(nonneg=True)
    b = cp.Parameter(nonneg=True)

    # NOT DPP — a * b is quadratic in parameters
    prob = cp.Problem(cp.Minimize(a * b * x), [x >= 1])
    print(prob.is_dpp())  # False

The fix is to compute the product in NumPy and pass it as a single parameter:

.. code:: python

    ab = cp.Parameter(nonneg=True)
    prob = cp.Problem(cp.Minimize(ab * x), [x >= 1])
    print(prob.is_dpp())  # True
    # Then set ab.value = a_val * b_val before each solve


.. _quadratic-objectives:

Use cp.sum_squares for quadratic objectives
--------------------------------------------

When your objective is a sum of squares, always use ``cp.sum_squares`` rather than
``cp.quad_form`` with an identity matrix. Using ``cp.quad_form(x, np.eye(n))`` constructs
a dense n×n matrix explicitly, which causes excessive memory usage and slow compile times
for large problems.

.. code:: python

    import cvxpy as cp
    import numpy as np

    n = 1000
    x = cp.Variable(n)

    # Slow and memory-intensive: constructs a dense 1000x1000 identity matrix
    objective = cp.Minimize(cp.quad_form(x, np.eye(n)))

    # Fast: purpose-built atom, no matrix construction
    objective = cp.Minimize(cp.sum_squares(x))

More generally, ``cp.quad_form(x, P)`` should only be used when ``P`` is a
non-trivial positive semidefinite matrix. Sparse matrices work perfectly fine
with ``cp.quad_form`` and do not cause the memory issues described above.
For diagonal ``P``, use ``cp.quad_form(x, scipy.sparse.diags_array(P_diag))`` or ``cp.sum(cp.multiply(P_diag, cp.square(x)))``
or restructure your problem to avoid it entirely.

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
  DPP-compliant problems with large parameters. Note that COO **requires** DPP
  compliance; if your problem is not DPP, the COO backend will silently fall
  back to the default backend.

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
| Avoid DPP-breaking patterns                  | High — silent fallback negates DPP benefits      |
+----------------------------------------------+--------------------------------------------------+
| Choose the right canonicalization backend    | Moderate — problem-dependent                     |
+----------------------------------------------+--------------------------------------------------+
| Use ``verbose=True`` to find bottlenecks     | Diagnostic — helps identify what to fix          |
+----------------------------------------------+--------------------------------------------------+

For a detailed benchmark, see the
`original notebook <https://github.com/cvxpy/cvxpy/blob/1.0/examples/notebooks/building_models_with_fast_compile_times.ipynb>`_
that inspired this page.
