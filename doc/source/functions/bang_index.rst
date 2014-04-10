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

Indexing and Slicing
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

Scalar Functions
----------------

These functions evaluate to a scalar value.

+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
|       Function      |        Meaning         |            Domain            |         Sign        |     Curvature     |        Monotonicity       |
+=====================+========================+==============================+=====================+===================+===========================+
| kl_div(x, y)        | :math:`x \log          | :math:`x, y > 0`             | !positive! positive | !convex! convex   | None                      |
|                     | (x/y)`                 |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| lamdba_max(X)       | :math:`\lambda_{       | :math:`X \in \mathbf{S}^n`   | !unknown! unknown   | !convex! convex   | None                      |
|                     | \max}(X)`              |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| lambda_min(X)       | :math:`\lambda_{       | :math:`X \in \mathbf{S}^n`   | !unknown! unknown   | !concave! concave | None                      |
|                     | \min}(X)`              |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| log_det(X)          | :math:`\log \det (X)`  | :math:`X \in \mathbf{S}^n_+` | !unknown! unknown   | !concave! concave | None                      |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| log_sum_exp(X)      | :math:`\log            | :math:`X \in                 | !unknown! unknown   | !convex! convex   | !incr! incr               |
|                     | \sum_{i,j}             | \mathbf{R}^{n \times m}`     |                     |                   |                           |
|                     | e^{X_{i,j}}`           |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X)             | :math:`\sqrt{          | :math:`X \in                 | !positive! positive | !convex! convex   | !incr! incr. for          |
|                     | \sum_{i,j}             | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{i,j} \geq 0`    |
| norm(X, 2)          | X_{i,j}^2 }`           |                              |                     |                   |                           |
|                     |                        |                              |                     |                   |                           |
| norm(X, "fro")      |                        |                              |                     |                   | !decr! decr. for          |
|                     |                        |                              |                     |                   | :math:`X_{i,j} \leq 0`    |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, 1)          | :math:`\sum_{i,j}      | :math:`X \in                 | !positive! positive | !convex! convex   | !incr! incr. for          |
|                     | \lvert X_{i,j} \rvert` | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{i,j} \geq 0`    |
|                     |                        |                              |                     |                   |                           |
|                     |                        |                              |                     |                   | !decr! decr. for          |
|                     |                        |                              |                     |                   | :math:`X_{i,j} \leq 0`    |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, "inf")      | :math:`\max_{i,j}      | :math:`X \in                 | !positive! positive | !convex! convex   | !incr! incr. for          |
|                     | \lvert X_{i,j} \rvert` | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{i,j} \geq 0`    |
|                     |                        |                              |                     |                   |                           |
|                     |                        |                              |                     |                   | !decr! decr. for          |
|                     |                        |                              |                     |                   | :math:`X_{i,j} \leq 0`    |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, "nuc")      | :math:`\mathrm{tr}     | :math:`X \in                 | !positive! positive | !convex! convex   | None                      |
|                     | \left(\sqrt{X^T X}     | \mathbf{R}^{n \times m}`     |                     |                   |                           |
|                     | \right)`               |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, "spec")     | :math:`\lambda_{\max}  | :math:`X \in                 | !positive! positive | !convex! convex   | None                      |
|                     | \left(\sqrt{X^T X}     | \mathbf{R}^{n \times m}`     |                     |                   |                           |
|                     | \right)`               |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_form(x, P)     | :math:`x^T P x`        | :math:`x \in \mathbf{R}^n`   | !positive! positive | !convex! convex   | !incr! incr. for          |
|                     |                        |                              |                     |                   | :math:`x_i \geq 0`        |
| P constant          |                        | :math:`P \in \mathbf{S}^n_+` |                     |                   |                           |
|                     |                        |                              |                     |                   | !decr! decr. for          |
|                     |                        |                              |                     |                   | :math:`x_i \leq 0`        |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_form(x, P)     | :math:`x^T P x`        | :math:`x \in \mathbf{R}^n`   | !negative! negative | !concave! concave | !decr! decr. for          |
|                     |                        |                              |                     |                   | :math:`x_i \geq 0`        |
| P constant          |                        | :math:`P \in \mathbf{S}^n_-` |                     |                   |                           |
|                     |                        |                              |                     |                   | !incr! incr. for          |
|                     |                        |                              |                     |                   | :math:`x_i \leq 0`        |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_form(c, X)     | :math:`c^T X c`        | :math:`c \in \mathbf{R}^n`   | depends on c, X     | !affine! affine   | depends on c              |
|                     |                        |                              |                     |                   |                           |
| c constant          |                        | :math:`X \in                 |                     |                   |                           |
|                     |                        | \mathbf{R}^{n \times n}`     |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_over_lin(x, y) | :math:`x^T x/y`        | :math:`x \in \mathbf{R}^n`   | !positive! positive | !convex! convex   | !decr! decr. for          |
|                     |                        |                              |                     |                   | :math:`x_i \geq 0`        |
|                     |                        | :math:`y > 0`                |                     |                   |                           |
|                     |                        |                              |                     |                   | !incr! incr. for          |
|                     |                        |                              |                     |                   | :math:`x_i \leq 0`        |
|                     |                        |                              |                     |                   |                           |
|                     |                        |                              |                     |                   | !decr! decr. in :math:`y` |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| sum(X)              | :math:`\sum_{i,j}      | :math:`X \in                 | depencs on X        | !affine! affine   | !incr! incr.              |
|                     | X_{i,j}`               | \mathbf{R}^{n \times m}`     |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+

Elementwise Functions
---------------------

These functions operate on each element of their arguments. For example, if ``X`` is a 5 by 4 matrix variable,
then ``abs(X)`` is a 5 by 4 matrix expression. ``abs(X)[1, 2]`` is equivalent to ``abs(X[1, 2])``.

Elementwise functions that take multiple arguments, such as ``max``, operate on the corresponding elements of each argument.
For example, if ``X`` and ``Y`` are both 3 by 3 matrix variables, then ``max(X, Y)`` is a 3 by 3 matrix expression.
``max(X)[2, 0]`` is equivalent to ``max(X[2, 0], Y[2, 0])``. This means all arguments must have the same dimensions or be
scalars, which are promoted.

+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
|     Function     |         Meaning         |           Domain           |          Sign          |     Curvature     |   Monotonicity   |
+==================+=========================+============================+========================+===================+==================+
| abs(x)           | :math:`\lvert x \rvert` | :math:`x \in \mathbf{R}`   | !positive! positive    | !convex! convex   | !incr! incr. for |
|                  |                         |                            |                        |                   | :math:`x \geq 0` |
|                  |                         |                            |                        |                   |                  |
|                  |                         |                            |                        |                   | !decr! decr. for |
|                  |                         |                            |                        |                   | :math:`x \leq 0` |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| entr(x)          | :math:`-x \log (x)`     | :math:`x > 0`              | !unknown! unknown      | !concave! concave | None             |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| exp(x)           | :math:`e^x`             | :math:`x \in \mathbf{R}`   | !positive! positive    | !convex! convex   | !incr! incr.     |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| inv_pos(x)       | :math:`1/x`             | :math:`x > 0`              | !positive! positive    | !convex! convex   | !decr! decr.     |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| log(x)           | :math:`\log(x)`         | :math:`x > 0`              | !unknown! unknown      | !concave! concave | !incr! incr.     |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| max(x1, ..., xk) | :math:`\max \left\{     | :math:`x_i \in \mathbf{R}` | max(sign(xi))          | !convex! convex   | !incr! incr.     |
|                  | x_1, \ldots , x_k       |                            |                        |                   |                  |
|                  | \right\}`               |                            |                        |                   |                  |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| min(x1, ..., xk) | :math:`\min \left\{     | :math:`x_i \in \mathbf{R}` | min(sign(xi))          | !concave! concave | !incr! incr.     |
|                  | x_1, \ldots , x_k       |                            |                        |                   |                  |
|                  | \right\}`               |                            |                        |                   |                  |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| neg(x)           | :math:`\max \left\{     | :math:`x \in \mathbf{R}`   | !positive! positive    | !convex! convex   | !decr! decr.     |
|                  | -x, 0 \right\}`         |                            |                        |                   |                  |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| pos(x)           | :math:`\max \left\{     | :math:`x \in \mathbf{R}`   | !positive! positive    | !convex! convex   | !incr! incr.     |
|                  | x, 0 \right\}`          |                            |                        |                   |                  |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| sqrt(x)          | :math:`\sqrt x`         | :math:`x \geq 0`           | !positive! positive    | !concave! concave | !incr! incr.     |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+
| square(x)        | :math:`x^2`             | :math:`x \in \mathbf{R}`   | !positive! positive    | !convex! convex   | !incr! incr. for |
|                  |                         |                            |                        |                   | :math:`x \geq 0` |
|                  |                         |                            |                        |                   |                  |
|                  |                         |                            |                        |                   | !decr! decr. for |
|                  |                         |                            |                        |                   | :math:`x \leq 0` |
+------------------+-------------------------+----------------------------+------------------------+-------------------+------------------+

Vector/Matrix Functions
-----------------------

These functions evaluate to a vector or matrix.

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
