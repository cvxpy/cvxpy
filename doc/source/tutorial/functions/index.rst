.. _functions:

Atomic Functions
================

This section of the tutorial describes the atomic functions that can be applied
to CVXPY expressions. CVXPY uses the function information in this section and
the :ref:`DCP rules <dcp>` to mark expressions with a sign and curvature.

Operators
---------

The infix operators ``+, -, *, /`` are treated as functions. ``+`` and
``-`` are affine functions. The expression ``expr1*expr2`` is are affine in
CVXPY when one of the expressions is constant, and ``expr1/expr2`` is affine
when ``expr2`` is a scalar constant.

Indexing and slicing
^^^^^^^^^^^^^^^^^^^^

Indexing in CVXPY follows exactly the same semantics as `NumPy ndarrays <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.
For example, if ``expr`` has shape ``(5,)`` then ``expr[1]`` gives the second entry.
More generally, ``expr[i:j:k]`` selects every kth
element of ``expr``, starting at ``i`` and ending at ``j-1``.
If ``expr`` is a matrix, then ``expr[i:j:k]`` selects rows,
while ``expr[i:j:k, r:s:t]`` selects both rows and columns.
Indexing drops dimensions while slicing preserves dimenions.
For example,

.. code:: python

     x = cvx.Variable(5)
     print("0 dimensional", x[0].shape)
     print("1 dimensional", x[0:1].shape)

::

    O dimensional: ()
    1 dimensional: (1,)

Transpose
^^^^^^^^^

The transpose of any expression can be obtained using the syntax
``expr.T``. Transpose is an affine function.

Power
^^^^^

For any CVXPY expression ``expr``,
the power operator ``expr**p`` is equivalent to
the function ``power(expr, p)``.

Scalar functions
----------------

A scalar function takes one or more scalars, vectors, or matrices as arguments
and returns a scalar.

.. |_| unicode:: 0xA0
   :trim:

.. list-table::
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Sign
     - Curvature |_|
     - Monotonicity

   * - :ref:`geo_mean(x) <geo-mean>`

       :ref:`geo_mean(x, p) <geo-mean>`

       :math:`p \in \mathbf{R}^n_{+}`

       :math:`p \neq 0`
     - :math:`x_1^{1/n} \cdots x_n^{1/n}`

       :math:`\left(x_1^{p_1} \cdots x_n^{p_n}\right)^{\frac{1}{\mathbf{1}^T p}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |positive| positive
     - |concave| concave
     - |incr| incr.

   * - :ref:`harmonic_mean(x) <harmonic-mean>`
     - :math:`\frac{n}{\frac{1}{x_1} + \cdots + \frac{1}{x_n}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |positive| positive
     - |concave| concave
     - |incr| incr.

   * - :ref:`lambda_max(X) <lambda-max>`
     - :math:`\lambda_{\max}(X)`
     - :math:`X \in \mathbf{S}^n`
     - |unknown| unknown
     - |convex| convex
     - None

   * - :ref:`lambda_min(X) <lambda-min>`
     - :math:`\lambda_{\min}(X)`
     - :math:`X \in \mathbf{S}^n`
     - |unknown| unknown
     - |concave| concave
     - None

   * - :ref:`lambda_sum_largest(X,k) <lambda-sum-largest>`

       :math:`k = 1,\ldots, n`
     - :math:`\text{sum of $k$ largest}\\ \text{eigenvalues of $X$}`
     - :math:`X \in\mathbf{S}^{n}`
     - |unknown| unknown
     - |convex| convex
     - None

   * - :ref:`lambda_sum_smallest(X,k) <lambda-sum-smallest>`

       :math:`k = 1,\ldots, n`
     - :math:`\text{sum of $k$ smallest}\\ \text{eigenvalues of $X$}`
     - :math:`X \in\mathbf{S}^{n}`
     - |unknown| unknown
     - |concave| concave
     - None

   * - :ref:`log_det(X) <log-det>`

     - :math:`\log \left(\det (X)\right)`
     - :math:`X \in \mathbf{S}^n_+`
     - |unknown| unknown
     - |concave| concave
     - None

   * - :ref:`log_sum_exp(X) <log-sum-exp>`

     - :math:`\log \left(\sum_{ij}e^{X_{ij}}\right)`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |unknown| unknown
     - |convex| convex
     - |incr| incr.

   * - :ref:`matrix_frac(x, P) <matrix-frac>`

     - :math:`x^T P^{-1} x`
     - :math:`x \in \mathbf{R}^n`

       :math:`P \in\mathbf{S}^n_{++}`
     - |positive| positive
     - |convex| convex
     - None

   * - :ref:`max(X) <max>`

     - :math:`\max_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |convex| convex
     - |incr| incr.

   * - :ref:`min(X) <min>`

     - :math:`\min_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |concave| concave
     - |incr| incr.

   * - :ref:`mixed_norm(X, p, q) <mixed-norm>`

     - :math:`\left(\sum_k\left(\sum_l\lvert x_{k,l}\rvert^p\right)^{q/p}\right)^{1/q}`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |positive| positive
     - |convex| convex
     - None

   * - :ref:`norm(x) <norm>`

       norm(x, 2)

     - :math:`\sqrt{\sum_{i} \lvert x_{i} \rvert^2 }`
     - :math:`X \in\mathbf{R}^{n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x_{i} \geq 0`

       |decr| for :math:`x_{i} \leq 0`

   * - :ref:`norm(X, "fro") <norm>`
     - :math:`\sqrt{\sum_{ij}X_{ij}^2 }`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`

   * - :ref:`norm(X, 1) <norm>`
     - :math:`\sum_{ij}\lvert X_{ij} \rvert`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`

   * - :ref:`norm(X, "inf") <norm>`
     - :math:`\max_{ij} \{\lvert X_{ij} \rvert\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`

   * - :ref:`norm(X, "nuc") <norm>`
     - :math:`\mathrm{tr}\left(\left(X^T X\right)^{1/2}\right)`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None

   * - :ref:`norm(X) <norm>`

       norm(X, 2)
     - :math:`\sqrt{\lambda_{\max}\left(X^T X\right)}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p \geq 1`

       or ``p = 'inf'``
     - :math:`\|X\|_p = \left(\sum_{ij} |X_{ij}|^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p < 1`, :math:`p \neq 0`
     - :math:`\|X\|_p = \left(\sum_{ij} X_{ij}^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_+`
     - |positive| positive
     - |concave| concave
     - |incr| incr.


   * - :ref:`quad_form(x, P) <quad-form>`

       constant :math:`P \in \mathbf{S}^n_+`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`


     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x_i \geq 0`

       |decr| for :math:`x_i \leq 0`

   * - :ref:`quad_form(x, P) <quad-form>`

       constant :math:`P \in \mathbf{S}^n_-`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`
     - |negative| negative
     - |concave| concave
     - |decr| for :math:`x_i \geq 0`

       |incr| for :math:`x_i \leq 0`

   * - :ref:`quad_form(c, X) <quad-form>`

       constant :math:`c \in \mathbf{R}^n`
     - :math:`c^T X c`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - depends |_| on |_| c, |_| X
     - |affine| affine
     - depends |_| on |_| c

   * - :ref:`quad_over_lin(X, y) <quad-over-lin>`

     - :math:`\left(\sum_{ij}X_{ij}^2\right)/y`
     - :math:`x \in \mathbf{R}^n`

       :math:`y > 0`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`

       |decr| decr. in :math:`y`

   * - :ref:`sum(X) <sum>`

     - :math:`\sum_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |affine| affine
     - |incr| incr.

   * - :ref:`sum_largest(X, k) <sum-largest>`

       :math:`k = 1,2,\ldots`
     - :math:`\text{sum of } k\text{ largest }X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |convex| convex
     - |incr| incr.

   * - :ref:`sum_smallest(X, k) <sum-smallest>`

       :math:`k = 1,2,\ldots`
     - :math:`\text{sum of } k\text{ smallest }X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |concave| concave
     - |incr| incr.

   * - :ref:`sum_squares(X) <sum-squares>`

     - :math:`\sum_{ij}X_{ij}^2`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`

   * - :ref:`trace(X) <trace>`

     - :math:`\mathrm{tr}\left(X \right)`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - same as X
     - |affine| affine
     - |incr| incr.

   * - :ref:`tv(x) <tv>`

     - :math:`\sum_{i}|x_{i+1} - x_i|`
     - :math:`x \in \mathbf{R}^n`
     - |positive| positive
     - |convex| convex
     - None

   * - :ref:`tv(x) <tv>`
     - :math:`\sum_{ij}\left\| \left[\begin{matrix} X_{i+1,j} - X_{ij} \\ X_{i,j+1} -X_{ij} \end{matrix}\right] \right\|_2`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None

   * - :ref:`tv([X1,...,Xk]) <tv>`
     - :math:`\sum_{ij}\left\| \left[\begin{matrix} X_{i+1,j}^{(1)} - X_{ij}^{(1)} \\ X_{i,j+1}^{(1)} -X_{ij}^{(1)} \\ \vdots \\ X_{i+1,j}^{(k)} - X_{ij}^{(k)} \\ X_{i,j+1}^{(k)} -X_{ij}^{(k)}  \end{matrix}\right] \right\|_2`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None

Clarifications
^^^^^^^^^^^^^^

The domain :math:`\mathbf{S}^n` refers to the set of symmetric matrices. The domains :math:`\mathbf{S}^n_+` and :math:`\mathbf{S}^n_-` refer to the set of positive semi-definite and negative semi-definite matrices, respectively. Similarly, :math:`\mathbf{S}^n_{++}` and :math:`\mathbf{S}^n_{--}` refer to the set of positive definite and negative definite matrices, respectively.

For a vector expression ``x``, ``norm(x)`` and ``norm(x, 2)`` give the Euclidean norm. For a matrix expression ``X``, however, ``norm(X)`` and ``norm(X, 2)`` give the spectral norm.

The function ``norm(X, "fro")`` is called the `Frobenius norm <http://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`__
and ``norm(X, "nuc")`` the `nuclear norm <http://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms>`__. The nuclear norm can also be defined as the sum of ``X``'s singular values.

The functions ``max`` and ``min`` give the largest and smallest entry, respectively, in a single expression. These functions should not be confused with ``maximum`` and ``minimum`` (see :ref:`elementwise`). Use ``maximum`` and ``minimum`` to find the max or min of a list of scalar expressions.

The CVXPY function ``sum`` sums all the entries in a single expression. The built-in Python ``sum`` should be used to add together a list of expressions. For example, the following code sums a list of three expressions:

.. code:: python

    expr_list = [expr1, expr2, expr3]
    expr_sum = sum(expr_list)


Functions along an axis
-----------------------

The functions ``sum``, ``norm``, ``max``, and ``min`` can be
applied along an axis.
Given an ``m`` by ``n`` expression ``expr``, the syntax ``func(expr, axis=0, keepdims=True)``
applies ``func`` to each column, returning a 1 by ``n`` expression.
The syntax ``func(expr, axis=1, keepdims=True)`` applies ``func`` to each row,
returning an ``m`` by 1 expression.
By default ``keepdims=False``, which means dimensions of length 1 are dropped.
For example, the following code sums
along the columns and rows of a matrix variable: 
.. code:: python

    X = cvx.Variable((5, 4))
    col_sums = cvx.sum(X, axis=0, keepdims=True) # Has size (1, 4)
    col_sums = cvx.sum(X, axis=0) # Has size (4,)
    row_sums = cvx.sum(X, axis=1) # Has size (5,)


.. _elementwise:

Elementwise functions
---------------------

These functions operate on each element of their arguments. For example, if ``X`` is a 5 by 4 matrix variable,
then ``abs(X)`` is a 5 by 4 matrix expression. ``abs(X)[1, 2]`` is equivalent to ``abs(X[1, 2])``.

Elementwise functions that take multiple arguments, such as ``maximum`` and ``multiply``, operate on the corresponding elements of each argument.
For example, if ``X`` and ``Y`` are both 3 by 3 matrix variables, then ``maximum(X, Y)`` is a 3 by 3 matrix expression.
``maximum(X, Y)[2, 0]`` is equivalent to ``maximum(X[2, 0], Y[2, 0])``. This means all arguments must have the same dimensions or be
scalars, which are promoted.

.. list-table::
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Sign
     - Curvature |_|
     - Monotonicity

   * - :ref:`abs(x) <abs>`

     - :math:`\lvert x \rvert`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`

   * - :ref:`entr(x) <entr>`

     - :math:`-x \log (x)`
     - :math:`x > 0`
     - |unknown| unknown
     - |concave| concave
     - None

   * - :ref:`exp(x) <exp>`

     - :math:`e^x`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| incr.

   * - :ref:`huber(x, M=1) <huber>`

       :math:`M \geq 0`
     - :math:`\begin{cases}x^2 &|x| \leq M  \\2M|x| - M^2&|x| >M\end{cases}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`

   * - :ref:`inv_pos(x) <inv-pos>`

     - :math:`1/x`
     - :math:`x > 0`
     - |positive| positive
     - |convex| convex
     - |decr| decr.

   * - :ref:`kl_div(x, y) <kl-div>`

     - :math:`x \log(x/y) - x + y`
     - :math:`x > 0`

       :math:`y > 0`
     - |positive| positive
     - |convex| convex
     - None

   * - :ref:`log(x) <log>`

     - :math:`\log(x)`
     - :math:`x > 0`
     - |unknown| unknown
     - |concave| concave
     - |incr| incr.

   * - :ref:`log1p(x) <log1p>`

     - :math:`\log(x+1)`
     - :math:`x > -1`
     - same as x
     - |concave| concave
     - |incr| incr.

   * - :ref:`logistic(x) <logistic>`

     - :math:`\log(1 + e^{x})`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| incr.

   * - :ref:`maximum(x, y) <maximum>`

     - :math:`\max \left\{x, y\right\}`
     - :math:`x,y \in \mathbf{R}`
     - depends on x,y 
     - |convex| convex
     - |incr| incr.

   * - :ref:`minimum(x, y) <minimum>`
     - :math:`\min \left\{x, y\right\}`
     - :math:`x, y \in \mathbf{R}`
     - depends |_| on |_| x,y
     - |concave| concave
     - |incr| incr.

   * - :ref:`multiply(c, x) <multiply>`

       :math:`c \in \mathbf{R}`
     - c*x
     - :math:`x \in\mathbf{R}`
     - :math:`\mathrm{sign}(cx)`
     - |affine| affine
     - depends |_| on |_| c

   * - :ref:`neg(x) <neg>`
     - :math:`\max \left\{-x, 0 \right\}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |decr| decr.

   * - :ref:`pos(x) <pos>`
     - :math:`\max \left\{x, 0 \right\}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| incr.

   * - :ref:`power(x, 0) <power>`
     - :math:`1`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - constant
     - |_|

   * - :ref:`power(x, 1) <power>`
     - :math:`x`
     - :math:`x \in \mathbf{R}`
     - same as x
     - |affine| affine
     - |incr| incr.

   * - :ref:`power(x, p) <power>`

       :math:`p = 2, 4, 8, \ldots`
     - :math:`x^p`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`

   * - :ref:`power(x, p) <power>`

       :math:`p < 0`
     - :math:`x^p`
     - :math:`x > 0`
     - |positive| positive
     - |convex| convex
     - |decr| decr.

   * - :ref:`power(x, p) <power>`

       :math:`0 < p < 1`
     - :math:`x^p`
     - :math:`x \geq 0`
     - |positive| positive
     - |concave| concave
     - |incr| incr.

   * - :ref:`power(x, p) <power>`

       :math:`p > 1,\ p \neq 2, 4, 8, \ldots`

     - :math:`x^p`
     - :math:`x \geq 0`
     - |positive| positive
     - |convex| convex
     - |incr| incr.

   * - :ref:`scalene(x, alpha, beta) <scalene>`

       :math:`\text{alpha} \geq 0`

       :math:`\text{beta} \geq 0`
     - :math:`\alpha\mathrm{pos}(x)+ \beta\mathrm{neg}(x)`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`

   * - :ref:`sqrt(x) <sqrt>`

     - :math:`\sqrt x`
     - :math:`x \geq 0`
     - |positive| positive
     - |concave| concave
     - |incr| incr.

   * - :ref:`square(x) <square>`

     - :math:`x^2`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`


Vector/matrix functions
-----------------------

A vector/matrix function takes one or more scalars, vectors, or matrices as arguments
and returns a vector or matrix.

.. list-table::
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Sign
     - Curvature |_|
     - Monotonicity

   * - :ref:`bmat([[X11,...,X1q],
       ...,
       [Xp1,...,Xpq]]) <bmat>`

     - :math:`\left[\begin{matrix} X^{(1,1)} &  \cdots &  X^{(1,q)} \\ \vdots &   & \vdots \\ X^{(p,1)} & \cdots &   X^{(p,q)} \end{matrix}\right]`
     - :math:`X^{(i,j)} \in\mathbf{R}^{m_i \times n_j}`
     - :math:`\mathrm{sign}\left(\sum_{ij} X^{(i,j)}_{11}\right)`
     - |affine| affine
     - |incr| incr.

   * - :ref:`conv(c, x) <conv>`

       :math:`c\in\mathbf{R}^m`
     - :math:`c*x`
     - :math:`x\in \mathbf{R}^n`
     - :math:`\mathrm{sign}\left(c_{1}x_{1}\right)`
     - |affine| affine
     - depends |_| on |_| c

   * - :ref:`cumsum(X, axis=0) <cumsum>`

     - cumulative sum along given axis.
     - :math:`X \in \mathbf{R}^{m \times n}`
     - same as X
     - |affine| affine
     - |incr| incr.

   * - :ref:`diag(x) <diag>`

     - :math:`\left[\begin{matrix}x_1  & &  \\& \ddots & \\& & x_n\end{matrix}\right]`
     - :math:`x \in\mathbf{R}^{n}`
     - same as x
     - |affine| affine
     - |incr| incr.

   * - :ref:`diag(X) <diag>`
     - :math:`\left[\begin{matrix}X_{11}  \\\vdots \\X_{nn}\end{matrix}\right]`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - same as X
     - |affine| affine
     - |incr| incr.

   * - :ref:`diff(X, k=1, axis=0) <diff>`

       :math:`k \in 0,1,2,\ldots`
     - kth order differences along given axis
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |affine| affine
     - |incr| incr.

   * - :ref:`hstack([X1, ..., Xk]) <hstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \cdots    X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n_i}`
     - :math:`\mathrm{sign}\left(\sum_i X^{(i)}_{11}\right)`
     - |affine| affine
     - |incr| incr.

   * - :ref:`kron(C, X) <kron>`

       :math:`C\in\mathbf{R}^{p \times q}`
     - :math:`\left[\begin{matrix}C_{11}X & \cdots & C_{1q}X \\ \vdots  &        & \vdots \\ C_{p1}X &  \cdots      & C_{pq}X     \end{matrix}\right]`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - :math:`\mathrm{sign}\left(C_{11}X_{11}\right)`
     - |affine| affine
     - depends |_| on C

   * - :ref:`reshape(X, (n', m')) <reshape>`

     - :math:`X' \in\mathbf{R}^{m' \times n'}`
     - :math:`X \in\mathbf{R}^{m \times n}`

       :math:`m'n' = mn`
     - same as X
     - |affine| affine
     - |incr| incr.

   * - :ref:`vec(X) <vec>`

     - :math:`x' \in\mathbf{R}^{mn}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |affine| affine
     - |incr| incr.

   * - :ref:`vstack([X1, ..., Xk]) <vstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \\ \vdots  \\X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m_i \times n}`
     - :math:`\mathrm{sign}\left(\sum_i X^{(i)}_{11}\right)`
     - |affine| affine
     - |incr| incr.


Clarifications
^^^^^^^^^^^^^^
The input to ``bmat`` is a list of lists of CVXPY expressions.
It constructs a block matrix.
The elements of each inner list are stacked horizontally and then the resulting block matrices are stacked vertically.

The output :math:`y` of ``conv(c, x)`` has size :math:`n+m-1` and is defined as
:math:`y[k]=\sum_{j=0}^k c[j]x[k-j]`.

The output :math:`x'` of ``vec(X)`` is the matrix :math:`X` flattened in column-major order into a vector.
Formally, :math:`x'_i = X_{i \bmod{m}, \left \lfloor{i/m}\right \rfloor }`.

The output :math:`X'` of ``reshape(X, (m', n'))`` is the matrix :math:`X` cast into an :math:`m' \times n'` matrix.
The entries are taken from :math:`X` in column-major order and stored in :math:`X'` in column-major order.
Formally, :math:`X'_{ij} = \mathbf{vec}(X)_{m'j + i}`.

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
