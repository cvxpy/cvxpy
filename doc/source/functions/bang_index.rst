.. _functions:

Functions
=========

This section of the tutorial describes the functions that can be applied
to CVXPY expressions. CVXPY uses the function information in this
section and the `DCP rules <dcp-rules>`__ to mark expressions with a
sign and curvature.

Operators
---------

The infix operators ``+, -, *, /`` are treated as functions. ``+`` and
``-`` are certainly affine functions. ``*`` and ``/`` are affine in
CVXPY because ``expr1*expr2`` is allowed only when one of the
expressions is constant and ``expr1/expr2`` is allowed only when
``expr2`` is a scalar constant.

Indexing and slicing
^^^^^^^^^^^^^^^^^^^^

All non-scalar expressions can be indexed using the syntax
``expr[i, j]``. Indexing is an affine function. The syntax ``expr[i]``
can be used as a shorthand for ``expr[i, 0]`` when ``expr`` is a column
vector. Similarly, ``expr[i]`` is shorthand for ``expr[0, i]`` when
``expr`` is a row vector.

Non-scalar Expressions can also be sliced into using the standard Python
slicing syntax. For example, ``expr[i:j:k, r]`` selects every kth
element in column r of ``expr``, starting at row i and ending at row
j-1.

Iteration
^^^^^^^^^

Expressions are iterable. Iterating over an expression returns indices
into the expression in column-major order. If ``expr`` is a 2 by 2
matrix, ``[elem for elem in expr]`` evaluates to
``[expr[0, 0], expr[1, 0], expr[0, 1], expr[1, 1]]``.

Transpose
^^^^^^^^^

The transpose of any expression can be obtained using the syntax
``expr.T``. Transpose is an affine function.

Scalar functions
----------------

A scalar function takes one or more scalars, vectors, or matrices as arguments
and returns a scalar.

+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
|       Function      |         Meaning          |            Domain            |         Sign        |     Curvature     |        Monotonicity       |
+=====================+==========================+==============================+=====================+===================+===========================+
| entr(X)             | :math:`\sum_{i,j}        | :math:`X_{i,j} > 0`          | !unknown! unknown   | !concave! concave | None                      |
|                     | -X_{i,j} \log (X_{i,j})` |                              |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| kl_div(X, Y)        | :math:`\sum_{i,j}        | :math:`X_{i,j} > 0`          | !positive! positive | !convex! convex   | None                      |
|                     | X_{i,j} \log(X_{i,j}     |                              |                     |                   |                           |
|                     | /Y_{i,j})`               | :math:`Y_{i,j} > 0`          |                     |                   |                           |
|                     |                          |                              |                     |                   |                           |
|                     | :math:`-X_{i,j}+Y_{i,j}` |                              |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| lamdba_max(X)       | :math:`\lambda_{         | :math:`X \in \mathbf{S}^n`   | !unknown! unknown   | !convex! convex   | None                      |
|                     | \max}(X)`                |                              |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| lambda_min(X)       | :math:`\lambda_{         | :math:`X \in \mathbf{S}^n`   | !unknown! unknown   | !concave! concave | None                      |
|                     | \min}(X)`                |                              |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| log_det(X)          | :math:`\log \left(       | :math:`X \in \mathbf{S}^n_+` | !unknown! unknown   | !concave! concave | None                      |
|                     | \det (X)\right)`         |                              |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| log_sum_exp(X)      | :math:`\log              | :math:`X \in                 | !unknown! unknown   | !convex! convex   | !incr! incr.              |
|                     | \sum_{i,j}               | \mathbf{R}^{n \times m}`     |                     |                   |                           |
|                     | e^{X_{i,j}}`             |                              |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(x)             | :math:`\sqrt{            | :math:`X \in                 | !positive! positive | !convex! convex   | !incr! for                |
|                     | \sum_{i}                 | \mathbf{R}^{n}`              |                     |                   | :math:`x_{i} \geq 0`      |
| norm(x, 2)          | x_{i}^2 }`               |                              |                     |                   |                           |
|                     |                          |                              |                     |                   |                           |
|                     |                          |                              |                     |                   | !decr! for                |
|                     |                          |                              |                     |                   | :math:`x_{i} \leq 0`      |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, "fro")      | :math:`\sqrt{            | :math:`X \in                 | !positive! positive | !convex! convex   | !incr! for                |
|                     | \sum_{i,j}               | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{i,j} \geq 0`    |
|                     | X_{i,j}^2 }`             |                              |                     |                   |                           |
|                     |                          |                              |                     |                   |                           |
|                     |                          |                              |                     |                   | !decr! for                |
|                     |                          |                              |                     |                   | :math:`X_{i,j} \leq 0`    |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, 1)          | :math:`\sum_{i,j}        | :math:`X \in                 | !positive! positive | !convex! convex   | !incr! for                |
|                     | \lvert X_{i,j} \rvert`   | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{i,j} \geq 0`    |
|                     |                          |                              |                     |                   |                           |
|                     |                          |                              |                     |                   | !decr! for                |
|                     |                          |                              |                     |                   | :math:`X_{i,j} \leq 0`    |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, "inf")      | :math:`\max_{i,j}        | :math:`X \in                 | !positive! positive | !convex! convex   | !incr! for                |
|                     | \lvert X_{i,j} \rvert`   | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{i,j} \geq 0`    |
|                     |                          |                              |                     |                   |                           |
|                     |                          |                              |                     |                   | !decr! for                |
|                     |                          |                              |                     |                   | :math:`X_{i,j} \leq 0`    |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, "nuc")      | :math:`\mathrm{tr}       | :math:`X \in                 | !positive! positive | !convex! convex   | None                      |
|                     | \left(\left(X^T X        | \mathbf{R}^{n \times m}`     |                     |                   |                           |
|                     | \right)^{1/2}\right)`    |                              |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X)             | :math:`\sqrt{            | :math:`X \in                 | !positive! positive | !convex! convex   | None                      |
|                     | \lambda_{\max}           | \mathbf{R}^{n \times m}`     |                     |                   |                           |
| norm(X, 2)          | \left(X^T X\right)}`     |                              |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_form(x, P)     | :math:`x^T P x`          | :math:`x \in \mathbf{R}^n`   | !positive! positive | !convex! convex   | !incr! for                |
|                     |                          |                              |                     |                   | :math:`x_i \geq 0`        |
| P constant          |                          | :math:`P \in \mathbf{S}^n_+` |                     |                   |                           |
|                     |                          |                              |                     |                   | !decr! for                |
|                     |                          |                              |                     |                   | :math:`x_i \leq 0`        |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_form(x, P)     | :math:`x^T P x`          | :math:`x \in \mathbf{R}^n`   | !negative! negative | !concave! concave | !decr! for                |
|                     |                          |                              |                     |                   | :math:`x_i \geq 0`        |
| P constant          |                          | :math:`P \in \mathbf{S}^n_-` |                     |                   |                           |
|                     |                          |                              |                     |                   | !incr! for                |
|                     |                          |                              |                     |                   | :math:`x_i \leq 0`        |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_form(c, X)     | :math:`c^T X c`          | :math:`c \in \mathbf{R}^n`   | depends on c, X     | !affine! affine   | depends on c              |
|                     |                          |                              |                     |                   |                           |
| c constant          |                          | :math:`X \in                 |                     |                   |                           |
|                     |                          | \mathbf{R}^{n \times n}`     |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_over_lin(X, y) | :math:`\sum_{i,j}        | :math:`x \in \mathbf{R}^n`   | !positive! positive | !convex! convex   | !incr! for                |
|                     | X_{i,j}^2/y`             |                              |                     |                   | :math:`X_{i,j} \geq 0`    |
|                     |                          | :math:`y > 0`                |                     |                   |                           |
|                     |                          |                              |                     |                   | !decr! for                |
|                     |                          |                              |                     |                   | :math:`X_{i,j} \leq 0`    |
|                     |                          |                              |                     |                   |                           |
|                     |                          |                              |                     |                   | !decr! decr. in :math:`y` |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+
| sum_entries(X)      | :math:`\sum_{i,j}        | :math:`X \in                 | depends on X        | !affine! affine   | !incr! incr.              |
|                     | X_{i,j}`                 | \mathbf{R}^{n \times m}`     |                     |                   |                           |
+---------------------+--------------------------+------------------------------+---------------------+-------------------+---------------------------+

Add min_entries, max_entries. Compare with max_elemwise, min_elemwise. Discuss norm(x,2) vector vs norm(X,2) matrix. Define Sn, Sn+, Sn-.
Add huber. Talk about sum_entries vs. built-in sum.

Elementwise functions
---------------------

These functions operate on each element of their arguments. For example, if ``X`` is a 5 by 4 matrix variable,
then ``abs(X)`` is a 5 by 4 matrix expression. ``abs(X)[1, 2]`` is equivalent to ``abs(X[1, 2])``.

Elementwise functions that take multiple arguments, such as ``max_elemwise``, operate on the corresponding elements of each argument.
For example, if ``X`` and ``Y`` are both 3 by 3 matrix variables, then ``max_elemwise(X, Y)`` is a 3 by 3 matrix expression.
``max_elemwise(X, Y)[2, 0]`` is equivalent to ``max_elemwise(X[2, 0], Y[2, 0])``. This means all arguments must have the same dimensions or be
scalars, which are promoted.

+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
|          Function         |         Meaning         |           Domain           |         Sign        |     Curvature     |   Monotonicity   |
+===========================+=========================+============================+=====================+===================+==================+
| abs(x)                    | :math:`\lvert x \rvert` | :math:`x \in \mathbf{R}`   | !positive! positive | !convex! convex   | !incr! for       |
|                           |                         |                            |                     |                   | :math:`x \geq 0` |
|                           |                         |                            |                     |                   |                  |
|                           |                         |                            |                     |                   | !decr! for       |
|                           |                         |                            |                     |                   | :math:`x \leq 0` |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| exp(x)                    | :math:`e^x`             | :math:`x \in \mathbf{R}`   | !positive! positive | !convex! convex   | !incr! incr.     |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| inv_pos(x)                | :math:`1/x`             | :math:`x > 0`              | !positive! positive | !convex! convex   | !decr! decr.     |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| log(x)                    | :math:`\log(x)`         | :math:`x > 0`              | !unknown! unknown   | !concave! concave | !incr! incr.     |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| max_elemwise(x1, ..., xk) | :math:`\max \left\{     | :math:`x_i \in \mathbf{R}` | max(sign(xi))       | !convex! convex   | !incr! incr.     |
|                           | x_1, \ldots , x_k       |                            |                     |                   |                  |
|                           | \right\}`               |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| min_elemwise(x1, ..., xk) | :math:`\min \left\{     | :math:`x_i \in \mathbf{R}` | min(sign(xi))       | !concave! concave | !incr! incr.     |
|                           | x_1, \ldots , x_k       |                            |                     |                   |                  |
|                           | \right\}`               |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| neg(x)                    | :math:`\max \left\{     | :math:`x \in \mathbf{R}`   | !positive! positive | !convex! convex   | !decr! decr.     |
|                           | -x, 0 \right\}`         |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| pos(x)                    | :math:`\max \left\{     | :math:`x \in \mathbf{R}`   | !positive! positive | !convex! convex   | !incr! incr.     |
|                           | x, 0 \right\}`          |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| sqrt(x)                   | :math:`\sqrt x`         | :math:`x \geq 0`           | !positive! positive | !concave! concave | !incr! incr.     |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| square(x)                 | :math:`x^2`             | :math:`x \in \mathbf{R}`   | !positive! positive | !convex! convex   | !incr! for       |
|                           |                         |                            |                     |                   | :math:`x \geq 0` |
|                           |                         |                            |                     |                   |                  |
|                           |                         |                            |                     |                   | !decr! for       |
|                           |                         |                            |                     |                   | :math:`x \leq 0` |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+

Change name of abs so doesn't conflict with built-in?

Vector/Matrix functions
-----------------------

A vector/matrix function takes one or more scalars, vectors, or matrices as arguments
and returns a vector or matrix.

+---------------------+-----------------------------+--------------------------+------------------------+-----------------+--------------+
|       Function      |           Meaning           |          Domain          |          Sign          |    Curvature    | Monotonicity |
+=====================+=============================+==========================+========================+=================+==============+
| vstack(x1, ..., xk) | :math:`\left[\begin{matrix} | :math:`x_i \in           | depends on :math:`x_i` | !affine! affine | !incr! incr. |
|                     | x_1  \\                     | \mathbf{R}^{n \times m}` |                        |                 |              |
|                     | \vdots  \\                  |                          |                        |                 |              |
|                     | x_k                         |                          |                        |                 |              |
|                     | \end{matrix}\right]`        |                          |                        |                 |              |
+---------------------+-----------------------------+--------------------------+------------------------+-----------------+--------------+

.. |positive| image:: functions_files/positive.svg
			  :width: 15px
			  :height: 15px

.. |negative| image:: functions_files/negative.svg
			  :width: 15px
			  :height: 15px

.. |unknown| image:: functions_files/unknown.svg
			  :width: 15px
			  :height: 15px

.. |convex| image:: functions_files/convex.svg
			  :width: 15px
			  :height: 15px

.. |concave| image:: functions_files/concave.svg
			  :width: 15px
			  :height: 15px

.. |affine| image:: functions_files/affine.svg
			  :width: 15px
			  :height: 15px

.. |incr| image:: functions_files/increasing.svg
			  :width: 15px
			  :height: 15px

.. |decr| image:: functions_files/decreasing.svg
			  :width: 15px
			  :height: 15px
