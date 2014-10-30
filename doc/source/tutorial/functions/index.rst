.. _functions:

Functions
=========

This section of the tutorial describes the functions that can be applied
to CVXPY expressions. CVXPY uses the function information in this
section and the `DCP rules <../dcp/index.html>`__ to mark expressions with a
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

Non-scalar expressions can also be sliced into using the standard Python
slicing syntax. For example, ``expr[i:j:k, r]`` selects every kth
element in column r of ``expr``, starting at row i and ending at row
j-1.

Transpose
^^^^^^^^^

The transpose of any expression can be obtained using the syntax
``expr.T``. Transpose is an affine function.

Scalar functions
----------------

A scalar function takes one or more scalars, vectors, or matrices as arguments
and returns a scalar.

+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
|       Function      |        Meaning         |            Domain            |         Sign        |     Curvature     |        Monotonicity       |
+=====================+========================+==============================+=====================+===================+===========================+
| entr(X)             | :math:`\sum_{ij}       | :math:`X_{ij} > 0`           | |unknown| unknown   | |concave| concave | None                      |
|                     | -X_{ij} \log (X_{ij})` |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| kl_div(X, Y)        | :math:`\sum_{ij}\left( | :math:`X_{ij} > 0`           | |positive| positive | |convex| convex   | None                      |
|                     | X_{ij} \log(X_{ij}     |                              |                     |                   |                           |
|                     | /Y_{ij}) \\            | :math:`Y_{ij} > 0`           |                     |                   |                           |
|                     | -X_{ij}+Y_{ij}         |                              |                     |                   |                           |
|                     | \right)`               |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| lamdba_max(X)       | :math:`\lambda_{       | :math:`X \in \mathbf{S}^n`   | |unknown| unknown   | |convex| convex   | None                      |
|                     | \max}(X)`              |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| lambda_min(X)       | :math:`\lambda_{       | :math:`X \in \mathbf{S}^n`   | |unknown| unknown   | |concave| concave | None                      |
|                     | \min}(X)`              |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| log_det(X)          | :math:`\log \left(     | :math:`X \in \mathbf{S}^n_+` | |unknown| unknown   | |concave| concave | None                      |
|                     | \det (X)\right)`       |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| log_sum_exp(X)      | :math:`\log \left(     | :math:`X \in                 | |unknown| unknown   | |convex| convex   | |incr| incr.              |
|                     | \sum_{ij}              | \mathbf{R}^{n \times m}`     |                     |                   |                           |
|                     | e^{X_{ij}}\right)`     |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| matrix_frac(x, P)   | :math:`x^T P^{-1} x`   | :math:`x \in \mathbf{R}^n`   | |positive| positive | |convex| convex   | None                      |
|                     |                        |                              |                     |                   |                           |
|                     |                        | :math:`P \in                 |                     |                   |                           |
|                     |                        | \mathbf{S}^n_{++}`           |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| max_entries(X)      | :math:`\max_{ij}       | :math:`X \in                 | same as X           | |convex| convex   | |incr| incr.              |
|                     | \left\{ X_{ij}         | \mathbf{R}^{n \times m}`     |                     |                   |                           |
|                     | \right\}`              |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| min_entries(X)      | :math:`\min_{ij}       | :math:`X \in                 | same as X           | |concave| concave | |incr| incr.              |
|                     | \left\{ X_{ij}         | \mathbf{R}^{n \times m}`     |                     |                   |                           |
|                     | \right\}`              |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| mixed_norm(X, p, q) | :math:`\left(\sum_k    | :math:`X \in                 | |positive| positive | |convex| convex   | None                      |
|                     | \left(\sum_l           | \mathbf{R}^{m,n}`            |                     |                   |                           |
|                     | \lvert x_{k,l}\rvert^p |                              |                     |                   |                           |
|                     | \right)^{q/p}          |                              |                     |                   |                           |
|                     | \right)^{1/q}`         |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(x)             | :math:`\sqrt{          | :math:`X \in                 | |positive| positive | |convex| convex   | |incr| for                |
|                     | \sum_{i}               | \mathbf{R}^{n}`              |                     |                   | :math:`x_{i} \geq 0`      |
| norm(x, 2)          | x_{i}^2 }`             |                              |                     |                   |                           |
|                     |                        |                              |                     |                   |                           |
|                     |                        |                              |                     |                   | |decr| for                |
|                     |                        |                              |                     |                   | :math:`x_{i} \leq 0`      |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, "fro")      | :math:`\sqrt{          | :math:`X \in                 | |positive| positive | |convex| convex   | |incr| for                |
|                     | \sum_{ij}              | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{ij} \geq 0`     |
|                     | X_{ij}^2 }`            |                              |                     |                   |                           |
|                     |                        |                              |                     |                   |                           |
|                     |                        |                              |                     |                   | |decr| for                |
|                     |                        |                              |                     |                   | :math:`X_{ij} \leq 0`     |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, 1)          | :math:`\sum_{ij}       | :math:`X \in                 | |positive| positive | |convex| convex   | |incr| for                |
|                     | \lvert X_{ij} \rvert`  | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{ij} \geq 0`     |
|                     |                        |                              |                     |                   |                           |
|                     |                        |                              |                     |                   | |decr| for                |
|                     |                        |                              |                     |                   | :math:`X_{ij} \leq 0`     |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, "inf")      | :math:`\max_{ij} \{    | :math:`X \in                 | |positive| positive | |convex| convex   | |incr| for                |
|                     | \lvert X_{ij} \rvert   | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{ij} \geq 0`     |
|                     | \}`                    |                              |                     |                   |                           |
|                     |                        |                              |                     |                   | |decr| for                |
|                     |                        |                              |                     |                   | :math:`X_{ij} \leq 0`     |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X, "nuc")      | :math:`\mathrm{tr}     | :math:`X \in                 | |positive| positive | |convex| convex   | None                      |
|                     | \left(\left(X^T X      | \mathbf{R}^{n \times m}`     |                     |                   |                           |
|                     | \right)^{1/2}\right)`  |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| norm(X)             | :math:`\sqrt{          | :math:`X \in                 | |positive| positive | |convex| convex   | None                      |
|                     | \lambda_{\max}         | \mathbf{R}^{n \times m}`     |                     |                   |                           |
| norm(X, 2)          | \left(X^T X\right)}`   |                              |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_form(x, P)     | :math:`x^T P x`        | :math:`x \in \mathbf{R}^n`   | |positive| positive | |convex| convex   | |incr| for                |
|                     |                        |                              |                     |                   | :math:`x_i \geq 0`        |
| P constant          |                        | :math:`P \in \mathbf{S}^n_+` |                     |                   |                           |
|                     |                        |                              |                     |                   | |decr| for                |
|                     |                        |                              |                     |                   | :math:`x_i \leq 0`        |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_form(x, P)     | :math:`x^T P x`        | :math:`x \in \mathbf{R}^n`   | |negative| negative | |concave| concave | |decr| for                |
|                     |                        |                              |                     |                   | :math:`x_i \geq 0`        |
| P constant          |                        | :math:`P \in \mathbf{S}^n_-` |                     |                   |                           |
|                     |                        |                              |                     |                   | |incr| for                |
|                     |                        |                              |                     |                   | :math:`x_i \leq 0`        |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_form(c, X)     | :math:`c^T X c`        | :math:`c \in \mathbf{R}^n`   | depends on c, X     | |affine| affine   | depends on c              |
|                     |                        |                              |                     |                   |                           |
| c constant          |                        | :math:`X \in                 |                     |                   |                           |
|                     |                        | \mathbf{R}^{n \times n}`     |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| quad_over_lin(X, y) | :math:`\left(\sum_{ij} | :math:`x \in \mathbf{R}^n`   | |positive| positive | |convex| convex   | |incr| for                |
|                     | X_{ij}^2\right)/y`     |                              |                     |                   | :math:`X_{ij} \geq 0`     |
|                     |                        | :math:`y > 0`                |                     |                   |                           |
|                     |                        |                              |                     |                   | |decr| for                |
|                     |                        |                              |                     |                   | :math:`X_{ij} \leq 0`     |
|                     |                        |                              |                     |                   |                           |
|                     |                        |                              |                     |                   | |decr| decr. in :math:`y` |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| sum_entries(X)      | :math:`\sum_{ij}       | :math:`X \in                 | same as X           | |affine| affine   | |incr| incr.              |
|                     | X_{ij}`                | \mathbf{R}^{n \times m}`     |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| sum_squares(X)      | :math:`\sum_{ij}       | :math:`X \in                 | |positive| positive | |convex| convex   | |incr| for                |
|                     | X_{ij}^2`              | \mathbf{R}^{n \times m}`     |                     |                   | :math:`X_{ij} \geq 0`     |
|                     |                        |                              |                     |                   |                           |
|                     |                        |                              |                     |                   | |decr| for                |
|                     |                        |                              |                     |                   | :math:`X_{ij} \leq 0`     |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+
| trace(X)            | :math:`\mathrm{tr}     | :math:`X \in                 | same as X           | |affine| affine   | |incr| incr.              |
|                     | \left(X \right)`       | \mathbf{R}^{n \times n}`     |                     |                   |                           |
+---------------------+------------------------+------------------------------+---------------------+-------------------+---------------------------+

Clarifications
^^^^^^^^^^^^^^

The domain :math:`\mathbf{S}^n` refers to the set of symmetric matrices. The domains :math:`\mathbf{S}^n_+` and :math:`\mathbf{S}^n_-` refer to the set of positive semi-definite and negative semi-definite matrices, respectively. Similarly, :math:`\mathbf{S}^n_{++}` and :math:`\mathbf{S}^n_{--}` refer to the set of positive definite and negative definite matrices, respectively.

For a vector expression ``x``, ``norm(x)`` and ``norm(x, 2)`` give the Euclidean norm. For a matrix expression ``X``, however, ``norm(X)`` and ``norm(X, 2)`` give the spectral norm.

The function ``norm(X, "fro")`` is called the `Frobenius norm <http://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`__
and ``norm(X, "nuc")`` the `nuclear norm <http://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms>`__. The nuclear norm can also be defined as the sum of ``X``'s singular values.

The functions ``max_entries`` and ``min_entries`` give the largest and smallest entry, respectively, in a single expression. These functions should not be confused with ``max_elemwise`` and ``min_elemwise`` (see :ref:`elementwise`). Use ``max_elemwise`` and ``min_elemwise`` to find the max or min of a list of scalar expressions.

The function ``sum_entries`` sums all the entries in a single expression. The built-in Python ``sum`` should be used to add together a list of expressions. For example, the following code sums the columns of a matrix variable:

.. code:: python

    X = Variable(100, 100)
    col_sum = sum([X[:, i] for i in range(X.size[1])])

.. _elementwise:

Elementwise functions
---------------------

These functions operate on each element of their arguments. For example, if ``X`` is a 5 by 4 matrix variable,
then ``abs(X)`` is a 5 by 4 matrix expression. ``abs(X)[1, 2]`` is equivalent to ``abs(X[1, 2])``.

Elementwise functions that take multiple arguments, such as ``max_elemwise`` and ``mul_elemwise``, operate on the corresponding elements of each argument.
For example, if ``X`` and ``Y`` are both 3 by 3 matrix variables, then ``max_elemwise(X, Y)`` is a 3 by 3 matrix expression.
``max_elemwise(X, Y)[2, 0]`` is equivalent to ``max_elemwise(X[2, 0], Y[2, 0])``. This means all arguments must have the same dimensions or be
scalars, which are promoted.

+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
|          Function         |         Meaning         |           Domain           |         Sign        |     Curvature     |   Monotonicity   |
+===========================+=========================+============================+=====================+===================+==================+
| abs(x)                    | :math:`\lvert x \rvert` | :math:`x \in \mathbf{R}`   | |positive| positive | |convex| convex   | |incr| for       |
|                           |                         |                            |                     |                   | :math:`x \geq 0` |
|                           |                         |                            |                     |                   |                  |
|                           |                         |                            |                     |                   | |decr| for       |
|                           |                         |                            |                     |                   | :math:`x \leq 0` |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| exp(x)                    | :math:`e^x`             | :math:`x \in \mathbf{R}`   | |positive| positive | |convex| convex   | |incr| incr.     |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| huber(x, M=1)             | :math:`\begin{cases}    | :math:`x \in \mathbf{R}`   | |positive| positive | |convex| convex   | |incr| for       |
|                           | x^2 &|x| \leq           |                            |                     |                   | :math:`x \geq 0` |
|                           | M  \\                   | :math:`M \geq 0`           |                     |                   |                  |
|                           | 2M|x| - M^2             |                            |                     |                   | |decr| for       |
|                           | &|x| >                  |                            |                     |                   | :math:`x \leq 0` |
|                           | M                       |                            |                     |                   |                  |
|                           | \end{cases}`            |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| inv_pos(x)                | :math:`1/x`             | :math:`x > 0`              | |positive| positive | |convex| convex   | |decr| decr.     |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| log(x)                    | :math:`\log(x)`         | :math:`x > 0`              | |unknown| unknown   | |concave| concave | |incr| incr.     |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| log1p(x)                  | :math:`\log(x+1)`       | :math:`x > -1`             | sign(x)             | |concave| concave | |incr| incr.     |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| max_elemwise(x1, ..., xk) | :math:`\max \left\{     | :math:`x_i \in \mathbf{R}` | max(sign(xi))       | |convex| convex   | |incr| incr.     |
|                           | x_1, \ldots , x_k       |                            |                     |                   |                  |
|                           | \right\}`               |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| min_elemwise(x1, ..., xk) | :math:`\min \left\{     | :math:`x_i \in \mathbf{R}` | min(sign(xi))       | |concave| concave | |incr| incr.     |
|                           | x_1, \ldots , x_k       |                            |                     |                   |                  |
|                           | \right\}`               |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| mul_elemwise(c, x)        | c*x                     | :math:`c,x \in             | sign(c*x)           | |affine| affine   | depends on c     |
|                           |                         | \mathbf{R}`                |                     |                   |                  |
| c constant                |                         |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| neg(x)                    | :math:`\max \left\{     | :math:`x \in \mathbf{R}`   | |positive| positive | |convex| convex   | |decr| decr.     |
|                           | -x, 0 \right\}`         |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| pos(x)                    | :math:`\max \left\{     | :math:`x \in \mathbf{R}`   | |positive| positive | |convex| convex   | |incr| incr.     |
|                           | x, 0 \right\}`          |                            |                     |                   |                  |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| scalene(x, alpha, beta)   | :math:`\alpha           | :math:`x \in \mathbf{R}`   | |positive| positive | |convex| convex   | |incr| for       |
|                           | \mathrm{pos}(x)         |                            |                     |                   | :math:`x \geq 0` |
| alpha >= 0                | + \beta                 |                            |                     |                   |                  |
|                           | \mathrm{neg}(x)`        |                            |                     |                   | |decr| for       |
| beta >= 0                 |                         |                            |                     |                   | :math:`x \leq 0` |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| sqrt(x)                   | :math:`\sqrt x`         | :math:`x \geq 0`           | |positive| positive | |concave| concave | |incr| incr.     |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+
| square(x)                 | :math:`x^2`             | :math:`x \in \mathbf{R}`   | |positive| positive | |convex| convex   | |incr| for       |
|                           |                         |                            |                     |                   | :math:`x \geq 0` |
|                           |                         |                            |                     |                   |                  |
|                           |                         |                            |                     |                   | |decr| for       |
|                           |                         |                            |                     |                   | :math:`x \leq 0` |
+---------------------------+-------------------------+----------------------------+---------------------+-------------------+------------------+

Vector/Matrix functions
-----------------------

A vector/matrix function takes one or more scalars, vectors, or matrices as arguments
and returns a vector or matrix.

+---------------------+-----------------------------+----------------------------+--------------------------+-----------------+--------------+
|       Function      |           Meaning           |           Domain           |           Sign           |    Curvature    | Monotonicity |
+=====================+=============================+============================+==========================+=================+==============+
| conv(c, x)          | :math:`c*x`                 | :math:`c\in\mathbf{R}^m`   | depends on c, x          | |affine| affine | depends on c |
|                     |                             |                            |                          |                 |              |
| c constant          |                             | :math:`x\in \mathbf{R}^n`  |                          |                 |              |
+---------------------+-----------------------------+----------------------------+--------------------------+-----------------+--------------+
| diag(x)             | :math:`\left[\begin{matrix} | :math:`x \in               | same as x                | |affine| affine | |incr| incr. |
|                     | x_1  & &  \\                | \mathbf{R}^{n}`            |                          |                 |              |
|                     | & \ddots & \\               |                            |                          |                 |              |
|                     | & & x_n                     |                            |                          |                 |              |
|                     | \end{matrix}\right]`        |                            |                          |                 |              |
+---------------------+-----------------------------+----------------------------+--------------------------+-----------------+--------------+
| diag(X)             | :math:`\left[\begin{matrix} | :math:`X \in               | same as X                | |affine| affine | |incr| incr. |
|                     | X_{11}  \\                  | \mathbf{R}^{n \times n}`   |                          |                 |              |
|                     | \vdots \\                   |                            |                          |                 |              |
|                     | X_{nn}                      |                            |                          |                 |              |
|                     | \end{matrix}\right]`        |                            |                          |                 |              |
+---------------------+-----------------------------+----------------------------+--------------------------+-----------------+--------------+
| hstack(X1, ..., Xk) | :math:`\left[\begin{matrix} | :math:`X_i \in             | sign(sum([x1, ..., xk])) | |affine| affine | |incr| incr. |
|                     | X_1  \cdots    X_k          | \mathbf{R}^{n \times m_i}` |                          |                 |              |
|                     | \end{matrix}\right]`        |                            |                          |                 |              |
+---------------------+-----------------------------+----------------------------+--------------------------+-----------------+--------------+
| reshape(X, n', m')  | :math:`X' \in               | :math:`X \in               | same as X                | |affine| affine | |incr| incr. |
|                     | \mathbf{R}^{n' \times m'}`  | \mathbf{R}^{n \times m}`   |                          |                 |              |
|                     |                             |                            |                          |                 |              |
|                     |                             | :math:`n'm' = nm`          |                          |                 |              |
+---------------------+-----------------------------+----------------------------+--------------------------+-----------------+--------------+
| vec(X)              | :math:`x' \in               | :math:`X \in               | same as X                | |affine| affine | |incr| incr. |
|                     | \mathbf{R}^{nm}`            | \mathbf{R}^{n \times m}`   |                          |                 |              |
|                     |                             |                            |                          |                 |              |
+---------------------+-----------------------------+----------------------------+--------------------------+-----------------+--------------+
| vstack(X1, ..., Xk) | :math:`\left[\begin{matrix} | :math:`X_i \in             | sign(sum([x1, ..., xk])) | |affine| affine | |incr| incr. |
|                     | X_1  \\                     | \mathbf{R}^{n_i \times m}` |                          |                 |              |
|                     | \vdots  \\                  |                            |                          |                 |              |
|                     | X_k                         |                            |                          |                 |              |
|                     | \end{matrix}\right]`        |                            |                          |                 |              |
+---------------------+-----------------------------+----------------------------+--------------------------+-----------------+--------------+

Clarifications
^^^^^^^^^^^^^^
The output :math:`y` of ``conv(c, x)`` has size :math:`n+m-1` and is defined as
:math:`y[k]=\sum_{j=0}^k c[j]x[k-j]`.

The output :math:`x'` of ``vec(X)`` is the matrix :math:`X` flattened in column-major order into a vector.
Formally, :math:`x'_i = X_{i \bmod{n}, \left \lfloor{i/n}\right \rfloor }`.

The output :math:`X'` of ``reshape(X, n', m')`` is the matrix :math:`X` cast into an :math:`n' \times m'` matrix.
The entries are taken from :math:`X` in column-major order and stored in :math:`X'` in column-major order.
Formally, :math:`X'_{ij} = \mathbf{vec}(X)_{n'j + i}`.

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
