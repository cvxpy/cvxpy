.. _matlab:

CVXPY for MATLAB and CVX users
==============================

Many CVXPY users arrive from MATLAB, and most of them arrive from `CVX
<https://cvxr.com/cvx/>`_. The modeling ideas carry over almost unchanged: you
declare variables, build an objective and a list of constraints, and hand the
problem to a solver. What does not carry over is the array semantics
underneath. In MATLAB every array is a matrix, and a plain list of numbers is a
row or column vector. In Python it is not, and CVXPY inherits NumPy's rules.

This page collects the differences that cause the most trouble. It assumes you
have read :ref:`the introductory tutorial <tutorial>`.

A problem side by side
----------------------

A bounded least-squares problem in CVX:

.. code:: matlab

    cvx_begin
        variable x(n)
        minimize( sum_square(A*x - b) )
        subject to
            0 <= x <= 1
    cvx_end

and in CVXPY:

.. code:: python

    import cvxpy as cp

    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()

Four differences are already visible:

* CVXPY has no ``cvx_begin``/``cvx_end`` block. Expressions are ordinary Python
  objects that you can build, store, and pass around.
* Constraints are a Python list, not statements in a block.
* ``0 <= x <= 1`` is **not** valid in CVXPY. Python evaluates chained
  comparisons in a way CVXPY cannot intercept, so you must write the two
  constraints separately. CVXPY raises an exception if you try.
* Matrix multiplication is ``@``, not ``*``. See
  :ref:`multiplication <matlab-multiplication>` below.

Four kinds of arrays
--------------------

Code that mixes MATLAB habits with CVXPY tends to confuse four distinct types:

``numpy.ndarray``
    The default numeric array in Python. Unlike a MATLAB matrix it can have any
    number of axes, including one. A NumPy array of shape ``(n,)`` is
    one-dimensional: it is neither a row nor a column vector.

SciPy sparse arrays
    Usable anywhere a dense constant is, including as the left operand of ``@``.
    CVXPY keeps them sparse internally, so prefer them for large structured data.

CVXPY ``Expression``
    A symbolic object: a :py:class:`~cvxpy.expressions.variable.Variable`, a
    :py:class:`~cvxpy.expressions.constants.parameter.Parameter`, a
    :py:class:`~cvxpy.expressions.constants.constant.Constant`, or anything built
    from them. All of them have a ``.shape``. A ``Constant`` always carries a
    numeric value and a ``Parameter`` carries one once assigned; a ``Variable``
    gets one from a solve, and keeps it afterwards.

MATLAB matrix
    Always two-dimensional, always column-major. Neither property holds in
    Python.

The rule of thumb: use NumPy arrays and SciPy sparse arrays for **data**, and
CVXPY expressions for **unknowns**. Mixing them in one expression is fine and
expected — ``A @ x + b`` with ``A`` sparse, ``b`` dense and ``x`` a variable
works.

.. _matlab-shapes:

Shapes and broadcasting
-----------------------

This is the single most common source of silent errors for MATLAB users.

In MATLAB, ``zeros(4,1)`` and a "vector of length 4" are the same thing. In
NumPy — and therefore in CVXPY — shapes ``(4,)``, ``(4, 1)`` and ``(1, 4)`` are
three different objects, and arithmetic between mismatched shapes does not fail:
it *broadcasts*.

.. code:: python

    import cvxpy as cp
    import numpy as np

    A = np.random.randn(4, 5)
    x = cp.Variable(5)

    (A @ x - np.zeros(4)).shape       # (4,)     — as intended
    (A @ x - np.zeros((4, 1))).shape  # (4, 4)   — a matrix, silently

The second line produces no error and no warning. The residual has become a
4-by-4 matrix, and any objective built on it optimizes something you did not
intend to write:

.. code:: python

    b = np.arange(4.)

    correct = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
    wrong   = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b.reshape(-1, 1))))

    correct.solve()   # 0.0
    wrong.solve()     # 20.0

Both problems are scalar-valued, both solve to optimality, and only one of them
is the intended least-squares problem. The only difference is the shape of ``b``.

The habit that prevents this: **do not create explicit row or column vectors in
Python unless something specifically requires one.** Write ``cp.Variable(n)``
rather than ``cp.Variable((n, 1))``, and keep data one-dimensional with
``b.ravel()`` when it arrives from a source that produced a column.

When you do need a block structure, build it explicitly with
:py:func:`~cvxpy.block` rather than relying on broadcasting to assemble it.

.. _matlab-multiplication:

Multiplication
--------------

MATLAB uses ``*`` for matrix multiplication and ``.*`` for elementwise. Python
splits this three ways:

============================  ==========================================
Operation                     CVXPY
============================  ==========================================
matrix-matrix, matrix-vector  ``A @ x``
scalar times anything         ``2 * x``
elementwise                   ``cp.multiply(a, x)``
============================  ==========================================

Using ``*`` for matrix multiplication has been deprecated since CVXPY 1.1 and
emits a warning. It is worth fixing rather than silencing: on operands that
happen to have equal shapes, the intent is genuinely ambiguous.

Reshaping: always pass ``order`` explicitly
-------------------------------------------

.. warning::

    The default order of :py:func:`~cvxpy.reshape`, :py:func:`~cvxpy.vec` and
    :py:func:`~cvxpy.flatten` is Fortran (``'F'``), and omitting ``order``
    raises a ``FutureWarning``. Switching the default to ``'C'`` to match NumPy
    is planned for CVXPY 2.0; until then the warning stays. **Pass** ``order``
    **explicitly** and the question does not arise.

MATLAB is column-major: ``reshape`` fills the first column, then the second.
NumPy is row-major by default. Name the order you want and CVXPY gives you that
one:

.. code:: python

    M = np.arange(6).reshape(2, 3)

    M.reshape(3, 2)                             # [[0, 1], [2, 3], [4, 5]]
    cp.reshape(cp.Constant(M), (3, 2), order='C').value  # [[0, 1], [2, 3], [4, 5]]
    cp.reshape(cp.Constant(M), (3, 2), order='F').value  # [[0, 4], [3, 2], [1, 5]]

Indexing
--------

Python indices start at ``0``, and slices exclude their endpoint, so MATLAB's
``x(1:3)`` is ``x[0:3]`` — same three entries, different notation. Transposition
is ``A.T``, not ``A'``.

.. code:: python

    M = cp.Variable((3, 4))

    M[0, 0].shape    # ()      scalar
    M[0].shape       # (4,)    first row, as a 1-D expression
    M[:, 0].shape    # (3,)    first column, also 1-D
    M[0:2].shape     # (2, 4)  first two rows

Note that indexing a single row or column drops the axis, exactly as in NumPy
and unlike MATLAB, where the result stays two-dimensional.

More than two dimensions
------------------------

Older material states that CVXPY expressions are limited to two dimensions.
That has not been true since CVXPY 1.6, which introduced
:ref:`N-dimensional expressions <n-dimensional>`, so ``cp.Variable((2, 3, 4))``
is valid today.

Coverage is broad: elementwise atoms, axis reductions, indexing, reshaping and
batched ``@`` all accept N-D input. A few atoms are still two-dimensional —
``tv`` and ``cvar`` among them — and gaps are tracked on the issue tracker.

Things CVXPY rejects that CVX accepts
-------------------------------------

* **Chained constraints.** ``0 <= x <= 1`` and ``x == y == 2`` raise an
  exception. Write them as separate entries in the constraint list.
* **Strict inequalities.** CVX accepts ``<`` and ``>``, but `interprets them
  identically to their non-strict counterparts
  <https://cvxr.com/cvx/doc/basics.html>`_. CVXPY raises an exception instead.
  The reason is that the resulting problem is ill-posed: an open feasible set
  need not attain its optimum. Numerical solvers also do not return strictly
  feasible points, so there would be no operational difference between the two
  anyway. Raising an error says so, rather than silently reinterpreting your
  model.
* **Non-symmetric quadratic forms.** :py:func:`~cvxpy.quad_form` requires a
  symmetric (or Hermitian) matrix and raises ``ValueError`` otherwise. If your
  matrix is symmetric only up to floating-point noise, symmetrize it first with
  ``(P + P.T) / 2``.

Getting results out
-------------------

CVX writes results into ``cvx_optval``, ``cvx_status`` and the variables
themselves. CVXPY attaches them to the objects you created:

=====================  ==================================
CVX                    CVXPY
=====================  ==================================
``cvx_optval``         ``prob.value``, also returned by ``prob.solve()``
``cvx_status``         ``prob.status``
value of ``x``         ``x.value`` — a NumPy ndarray
dual variable          ``constraint.dual_value``
=====================  ==================================

``x.value`` is ``None`` until the problem has been solved, and stays ``None`` if
the problem is infeasible or unbounded. Its shape matches the variable's shape,
so a ``cp.Variable(5)`` yields an array of shape ``(5,)`` — one-dimensional
again, not a column.

Where to go next
----------------

* :ref:`functions` — the atom library, with the CVX-equivalent names.
* :ref:`dcp` — the ruleset for what counts as a valid convex expression. It is
  the same discipline CVX enforces, and the error messages are more informative.
* :ref:`advanced` — solver selection and problem transformations.
