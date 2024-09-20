.. _functions:

Atomic Functions
================

This section of the tutorial describes the atomic functions that can be applied
to CVXPY expressions. CVXPY uses the function information in this section and
the :ref:`DCP rules <dcp>` to mark expressions with a sign and curvature.

Operators
---------

The infix operators ``+, -, *, /, @`` are treated as functions. The operators ``+`` and
``-`` are always affine functions. The expression ``expr1*expr2`` is affine in
CVXPY when one of the expressions is constant, and ``expr1/expr2`` is affine
when ``expr2`` is a scalar constant.

Historically, CVXPY used ``expr1 * expr2`` to denote matrix multiplication.
This is now deprecated. Starting with Python 3.5, users can write
``expr1 @ expr2`` for matrix multiplication and dot products.
As of CVXPY version 1.1, we are adopting a new standard:

* ``@`` should be used for matrix-matrix and matrix-vector multiplication,
* ``*`` should be matrix-scalar and vector-scalar multiplication

Elementwise multiplication can be applied with the :ref:`multiply` function.


Indexing and slicing
^^^^^^^^^^^^^^^^^^^^

Indexing in CVXPY follows exactly the same semantics as `NumPy ndarrays <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.
For example, if ``expr`` has shape ``(5,)`` then ``expr[1]`` gives the second entry.
More generally, ``expr[i:j:k]`` selects every kth
element of ``expr``, starting at ``i`` and ending at ``j-1``.
If ``expr`` is a matrix, then ``expr[i:j:k]`` selects rows,
while ``expr[i:j:k, r:s:t]`` selects both rows and columns.
Indexing drops dimensions while slicing preserves dimensions.
For example,

.. code:: python

     x = cvxpy.Variable(5)
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
     - DCP Properties
     - Curvature |_|

   * - :ref:`dotsort(X,W) <dotsort>`

       constant :math:`W \in \mathbf{R}^{o \times p}`
     - :math:`\text{dot product of}`
       :math:`\operatorname{sort}\operatorname{vec}(X) \text{ and}`
       :math:`\operatorname{sort}\operatorname{vec}(W)`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - sign depends on 
     
       :math:`X` and :math:`W`

       |incr| for :math:`\min(W) \geq 0`

       |decr| for :math:`\max(W) \leq 0`
     - |convex| convex

   * - :ref:`geo_mean(x) <geo-mean>`

       :ref:`geo_mean(x, p) <geo-mean>`

       :math:`p \in \mathbf{R}^n_{+}`

       :math:`p \neq 0`
     - :math:`x_1^{1/n} \cdots x_n^{1/n}`

       :math:`\left(x_1^{p_1} \cdots x_n^{p_n}\right)^{\frac{1}{\mathbf{1}^T p}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |positive| positive

       |incr| incr.
     - |concave| concave

   * - :ref:`harmonic_mean(x) <harmonic-mean>`
     - :math:`\frac{n}{\frac{1}{x_1} + \cdots + \frac{1}{x_n}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |positive| positive

       |incr| incr.
     - |concave| concave

   * - :ref:`inv_prod(x) <inv-prod>`
     - :math:`(x_1\cdots x_n)^{-1}`
     - :math:`x \in \mathbf{R}^n_+`
     - |positive| positive

       |decr| decr.
     - |convex| convex


   * - :ref:`lambda_max(X) <lambda-max>`
     - :math:`\lambda_{\max}(X)`
     - :math:`X \in \mathbf{S}^n`
     - |unknown| unknown sign
     - |convex| convex

   * - :ref:`lambda_min(X) <lambda-min>`
     - :math:`\lambda_{\min}(X)`
     - :math:`X \in \mathbf{S}^n`
     - |unknown| unknown sign
     - |concave| concave

   * - :ref:`lambda_sum_largest(X,k) <lambda-sum-largest>`

       :math:`k = 1,\ldots, n`
     - :math:`\text{sum of $k$ largest}`
       :math:`\text{eigenvalues of $X$}`
     - :math:`X \in\mathbf{S}^{n}`
     - |unknown| unknown sign
     - |convex| convex

   * - :ref:`lambda_sum_smallest(X,k) <lambda-sum-smallest>`

       :math:`k = 1,\ldots, n`
     - :math:`\text{sum of $k$ smallest}`
       :math:`\text{eigenvalues of $X$}`
     - :math:`X \in\mathbf{S}^{n}`
     - |unknown| unknown sign
     - |concave| concave

   * - :ref:`log_det(X) <log-det>`

     - :math:`\log \left(\det (X)\right)`
     - :math:`X \in \mathbf{S}^n_+`
     - |unknown| unknown sign
     - |concave| concave

   * - :ref:`log_sum_exp(X) <log-sum-exp>`

     - :math:`\log \left(\sum_{ij}e^{X_{ij}}\right)`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |unknown| unknown sign

       |incr| incr.
     - |convex| convex

   * - :ref:`matrix_frac(x, P) <matrix-frac>`

     - :math:`x^T P^{-1} x`
     - :math:`x \in \mathbf{R}^n`

       :math:`P \in\mathbf{S}^n_{++}`
     - |positive| positive
     - |convex| convex

   * - :ref:`max(X) <max>`

     - :math:`\max_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |convex| convex

   * - :ref:`mean(X) <mean>`

     - :math:`\frac{1}{m n}\sum_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |affine| affine

   * - :ref:`min(X) <min>`

     - :math:`\min_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |concave| concave

   * - :ref:`mixed_norm(X, p, q) <mixed-norm>`

     - :math:`\left(\sum_k\left(\sum_l\lvert x_{k,l}\rvert^p\right)^{q/p}\right)^{1/q}`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |positive| positive
     - |convex| convex

   * - :ref:`norm(x) <norm>`

       norm(x, 2)

     - :math:`\sqrt{\sum_{i} \lvert x_{i} \rvert^2 }`
     - :math:`X \in\mathbf{R}^{n}`
     - |positive| positive

       |incr| for :math:`x_{i} \geq 0`

       |decr| for :math:`x_{i} \leq 0`
     - |convex| convex

   * - :ref:`norm(x, 1) <norm>`
     - :math:`\sum_{i}\lvert x_{i} \rvert`
     - :math:`x \in\mathbf{R}^{n}`
     - |positive| positive

       |incr| for :math:`x_{i} \geq 0`

       |decr| for :math:`x_{i} \leq 0`
     - |convex| convex

   * - :ref:`norm(x, "inf") <norm>`
     - :math:`\max_{i} \{\lvert x_{i} \rvert\}`
     - :math:`x \in\mathbf{R}^{n}`
     - |positive| positive

       |incr| for :math:`x_{i} \geq 0`

       |decr| for :math:`x_{i} \leq 0`
     - |convex| convex

   * - :ref:`norm(X, "fro") <norm>`
     - :math:`\sqrt{\sum_{ij}X_{ij}^2 }`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive

       |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`
     - |convex| convex

   * - :ref:`norm(X, 1) <norm>`
     - :math:`\max_{j} \|X_{:,j}\|_1`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive

       |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`
     - |convex| convex

   * - :ref:`norm(X, "inf") <norm>`
     - :math:`\max_{i} \|X_{i,:}\|_1`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
       
       |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`
     - |convex| convex

   * - :ref:`norm(X, "nuc") <norm>`
     - :math:`\mathrm{tr}\left(\left(X^T X\right)^{1/2}\right)`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex

   * - :ref:`norm(X) <norm>`

       norm(X, 2)
     - :math:`\sqrt{\lambda_{\max}\left(X^T X\right)}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex

   * - :ref:`perspective(f(x),s) <perspective>`

     - :math:`sf(x/s)`
     - :math:`x \in \mathop{\bf dom} f`

       :math:`s \geq 0`
     - same sign as f
     - |convex| / |concave|

       same as :math:`f`

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p \geq 1`

       or ``p = 'inf'``
     - :math:`\left(\sum_{ij} |X_{ij}|^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
       
       |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`
     - |convex| convex

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p < 1`, :math:`p \neq 0`
     - :math:`\left(\sum_{ij} X_{ij}^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_+`
     - |positive| positive

       |incr| incr.
     - |concave| concave

   * - :ref:`ptp(X) <ptp>`

     - :math:`\max_{ij} X_{ij} - \min_{ij} X_{ij}`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex

   * - :ref:`quad_form(x, P) <quad-form>`

       constant :math:`P \in \mathbf{S}^n_+`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`
     - |positive| positive
       
       |incr| for :math:`x_i \geq 0`

       |decr| for :math:`x_i \leq 0`
     - |convex| convex

   * - :ref:`quad_form(x, P) <quad-form>`

       constant :math:`P \in \mathbf{S}^n_-`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`
     - |negative| negative
       
       |decr| for :math:`x_i \geq 0`

       |incr| for :math:`x_i \leq 0`
     - |concave| concave

   * - :ref:`quad_form(c, X) <quad-form>`

       constant :math:`c \in \mathbf{R}^n`
     - :math:`c^T X c`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - sign depends |_| on |_| c, |_| X
      
       monotonicity depends |_| on |_| c
     - |affine| affine

   * - :ref:`quad_over_lin(X, y) <quad-over-lin>`

     - :math:`\left(\sum_{ij}X_{ij}^2\right)/y`
     - :math:`x \in \mathbf{R}^n`

       :math:`y > 0`
     - |positive| positive
       
       |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`

       |decr| decr. in :math:`y`
     - |convex| convex

   * - :ref:`std(X) <std>`

     - analog to `numpy.std <https://numpy.org/doc/stable/reference/generated/numpy.std.html#numpy-std>`_
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex

   * - :ref:`sum(X) <sum>`

     - :math:`\sum_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |affine| affine

   * - :ref:`sum_largest(X, k) <sum-largest>`

       :math:`k = 1,2,\ldots`
     - :math:`\text{sum of } k\text{ largest }X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |convex| convex

   * - :ref:`sum_smallest(X, k) <sum-smallest>`

       :math:`k = 1,2,\ldots`
     - :math:`\text{sum of } k\text{ smallest }X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |concave| concave

   * - :ref:`sum_squares(X) <sum-squares>`

     - :math:`\sum_{ij}X_{ij}^2`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive

       |incr| for :math:`X_{ij} \geq 0`

       |decr| for :math:`X_{ij} \leq 0`
     - |convex| convex

   * - :ref:`trace(X) <trace>`

     - :math:`\mathrm{tr}\left(X \right)`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - same sign as X

       |incr| incr.
     - |affine| affine

   * - :ref:`tr_inv(X) <tr_inv>`

     - :math:`\mathrm{tr}\left(X^{-1} \right)`
     - :math:`X \in\mathbf{S}^n_{++}`
     - |positive| positive
     - |convex| convex

   * - :ref:`tv(x) <tv>`

     - :math:`\sum_{i}|x_{i+1} - x_i|`
     - :math:`x \in \mathbf{R}^n`
     - |positive| positive
     - |convex| convex

   * - :ref:`tv(X) <tv>`
       :math:`Y = \left[\begin{matrix} X_{i+1,j} - X_{ij} \\ X_{i,j+1} -X_{ij} \end{matrix}\right]`
     - :math:`\sum_{ij}\left\| Y \right\|_2`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex

   * - :ref:`tv([X1,...,Xk]) <tv>`
       :math:`Y = \left[\begin{matrix} X_{i+1,j}^{(1)} - X_{ij}^{(1)} \\ X_{i,j+1}^{(1)} -X_{ij}^{(1)} \\ \vdots \\ X_{i+1,j}^{(k)} - X_{ij}^{(k)} \\ X_{i,j+1}^{(k)} -X_{ij}^{(k)}  \end{matrix}\right]`
     - :math:`\sum_{ij}\left\| Y \right\|_2`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex

   * - :ref:`var(X) <var>`

     - analog to `numpy.var <https://numpy.org/doc/stable/reference/generated/numpy.var.html#numpy-var>`_
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex

   * - :ref:`von_neumann_entr(X) <von-neumann-entr>`
     - :math:`-\operatorname{tr}(X\operatorname{logm}(X))`
     - :math:`X \in \mathbf{S}^{n}_+`
     - |unknown| unknown sign
     - |concave| concave

Clarifications for scalar functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The domain :math:`\mathbf{S}^n` refers to the set of symmetric matrices. The domains :math:`\mathbf{S}^n_+` and :math:`\mathbf{S}^n_-` refer to the set of positive semi-definite and negative semi-definite matrices, respectively. Similarly, :math:`\mathbf{S}^n_{++}` and :math:`\mathbf{S}^n_{--}` refer to the set of positive definite and negative definite matrices, respectively.

For a vector expression ``x``, ``norm(x)`` and ``norm(x, 2)`` give the Euclidean norm. For a matrix expression ``X``, however, ``norm(X)`` and ``norm(X, 2)`` give the spectral norm.

The function ``norm(X, "fro")`` is called the `Frobenius norm <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`__
and ``norm(X, "nuc")`` the `nuclear norm <https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms>`__. The nuclear norm can also be defined as the sum of ``X``'s singular values.

The functions ``max`` and ``min`` give the largest and smallest entry, respectively, in a single expression. These functions should not be confused with ``maximum`` and ``minimum`` (see :ref:`elementwise`). Use ``maximum`` and ``minimum`` to find the max or min of a list of scalar expressions.

The CVXPY function ``sum`` sums all the entries in a single expression. The built-in Python ``sum`` should be used to add together a list of expressions. For example, the following code sums a list of three expressions:

.. code:: python

    expr_list = [expr1, expr2, expr3]
    expr_sum = sum(expr_list)


Functions along an axis
-----------------------

The functions ``sum``, ``norm``, ``max``, ``min``, ``mean``, ``std``, ``var``, and ``ptp`` can
be applied along an axis.
Given an ``m`` by ``n`` expression ``expr``, the syntax ``func(expr, axis=0, keepdims=True)``
applies ``func`` to each column, returning a 1 by ``n`` expression.
The syntax ``func(expr, axis=1, keepdims=True)`` applies ``func`` to each row,
returning an ``m`` by 1 expression.
By default ``keepdims=False``, which means dimensions of length 1 are dropped.
For example, the following code sums
along the columns and rows of a matrix variable:

.. code:: python

    X = cvxpy.Variable((5, 4))
    col_sums = cvxpy.sum(X, axis=0, keepdims=True) # Has size (1, 4)
    col_sums = cvxpy.sum(X, axis=0) # Has size (4,)
    row_sums = cvxpy.sum(X, axis=1) # Has size (5,)


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
     - DCP Properties
     - Curvature |_|

   * - :ref:`abs(x) <abs>`

     - :math:`\lvert x \rvert`
     - :math:`x \in \mathbf{C}`
     - |positive| positive
       
       |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`
     - |convex| convex

   * - :ref:`conj(x) <conj>`

     - complex conjugate
     - :math:`x \in \mathbf{C}`
     - |unknown| unknown sign
     - |affine| affine

   * - :ref:`entr(x) <entr>`

     - :math:`-x \log (x)`
     - :math:`x > 0`
     - |unknown| unknown sign
     - |concave| concave

   * - :ref:`exp(x) <exp>`

     - :math:`e^x`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
       
       |incr| incr.
     - |convex| convex

   * - :ref:`huber(x, M=1) <huber>`

       :math:`M \geq 0`
     - :math:`\begin{cases}x^2 &|x| \leq M  \\2M|x| - M^2&|x| >M\end{cases}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
       
       |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`
     - |convex| convex


   * - :ref:`imag(x) <imag-atom>`

     - imaginary part of a complex number
     - :math:`x \in \mathbf{C}`
     - |unknown| unknown sign
     - |affine| affine

   * - :ref:`inv_pos(x) <inv-pos>`

     - :math:`1/x`
     - :math:`x > 0`
     - |positive| positive

       |decr| decr.
     - |convex| convex

   * - :ref:`kl_div(x, y) <kl-div>`

     - :math:`x \log(x/y) - x + y`
     - :math:`x > 0`

       :math:`y > 0`
     - |positive| positive
     - |convex| convex

   * - :ref:`log(x) <log>`

     - :math:`\log(x)`
     - :math:`x > 0`
     - |unknown| unknown sign
       
       |incr| incr.
     - |concave| concave

   * - :ref:`log_normcdf(x) <log-normcdf>`

     - :ref:`approximate <clarifyelementwise>` log of the standard normal CDF
     - :math:`x \in \mathbf{R}`
     - |negative| negative

       |incr| incr.
     - |concave| concave

   * - :ref:`log1p(x) <log1p>`

     - :math:`\log(x+1)`
     - :math:`x > -1`
     - same sign as x

       |incr| incr.
     - |concave| concave

   * - :ref:`loggamma(x) <loggamma>`

     - :ref:`approximate <clarifyelementwise>` `log of the Gamma function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loggamma.html>`_
     - :math:`x > 0`
     - |unknown| unknown sign
     - |convex| convex

   * - :ref:`logistic(x) <logistic>`

     - :math:`\log(1 + e^{x})`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
       
       |incr| incr.
     - |convex| convex

   * - :ref:`maximum(x, y) <maximum>`

     - :math:`\max \left\{x, y\right\}`
     - :math:`x,y \in \mathbf{R}`
     - sign depends on x,y
       
       |incr| incr.
     - |convex| convex

   * - :ref:`minimum(x, y) <minimum>`
     - :math:`\min \left\{x, y\right\}`
     - :math:`x, y \in \mathbf{R}`
     - sign depends |_| on |_| x,y
       
       |incr| incr.
     - |concave| concave

   * - :ref:`multiply(c, x) <multiply>`

       :math:`c \in \mathbf{R}`
     - c*x
     - :math:`x \in\mathbf{R}`
     - :math:`\mathrm{sign}(cx)`
      
       monotonicity depends |_| on |_| c
     - |affine| affine

   * - :ref:`neg(x) <neg>`
     - :math:`\max \left\{-x, 0 \right\}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
       
       |decr| decr.
     - |convex| convex

   * - :ref:`pos(x) <pos>`
     - :math:`\max \left\{x, 0 \right\}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
       
       |incr| incr.
     - |convex| convex

   * - :ref:`power(x, 0) <power>`
     - :math:`1`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - constant

   * - :ref:`power(x, 1) <power>`
     - :math:`x`
     - :math:`x \in \mathbf{R}`
     - same sign as x
       
       |incr| incr.
     - |affine| affine

   * - :ref:`power(x, p) <power>`

       :math:`p = 2, 4, 8, \ldots`
     - :math:`x^p`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
       
       |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`
     - |convex| convex

   * - :ref:`power(x, p) <power>`

       :math:`p < 0`
     - :math:`x^p`
     - :math:`x > 0`
     - |positive| positive
       
       |decr| decr.
     - |convex| convex

   * - :ref:`power(x, p) <power>`

       :math:`0 < p < 1`
     - :math:`x^p`
     - :math:`x \geq 0`
     - |positive| positive
       
       |incr| incr.
     - |concave| concave

   * - :ref:`power(x, p) <power>`

       :math:`p > 1,\ p \neq 2, 4, 8, \ldots`

     - :math:`x^p`
     - :math:`x \geq 0`
     - |positive| positive
       
       |incr| incr.
     - |convex| convex

   * - :ref:`real(x) <real-atom>`

     - real part of a complex number
     - :math:`x \in \mathbf{C}`
     - |unknown| unknown
       
       |incr| incr.
     - |affine| affine

   * - :ref:`rel_entr(x, y) <rel-entr>`

     - :math:`x \log(x/y)`
     - :math:`x > 0`

       :math:`y > 0`
     - |unknown| unknown sign
       
       |decr| in :math:`y`
     - |convex| convex

   * - :ref:`scalene(x, alpha, beta) <scalene>`

       :math:`\text{alpha} \geq 0`

       :math:`\text{beta} \geq 0`
     - :math:`\alpha\mathrm{pos}(x)+ \beta\mathrm{neg}(x)`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
       
       |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`
     - |convex| convex

   * - :ref:`sqrt(x) <sqrt>`

     - :math:`\sqrt x`
     - :math:`x \geq 0`
     - |positive| positive
       
       |incr| incr.
     - |concave| concave

   * - :ref:`square(x) <square>`

     - :math:`x^2`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
      
       |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`
     - |convex| convex

   * - :ref:`xexp(x) <xexp>`

     - :math:`x e^x`
     - :math:`x \geq 0`
     - |positive| positive
       
       |incr| incr.
     - |convex| convex

.. _clarifyelementwise:

Clarifications on elementwise functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The functions ``log_normcdf`` and ``loggamma`` are defined via approximations. ``log_normcdf`` has highest accuracy
over the range -4 to 4, while ``loggamma`` has similar accuracy over all positive reals.
See `CVXPY GitHub PR #1224 <https://github.com/cvxpy/cvxpy/pull/1224#issue-793221374>`_
and `CVXPY GitHub Issue #228 <https://github.com/cvxpy/cvxpy/issues/228#issuecomment-544281906>`_
for details on the approximations.

Vector/matrix functions
-----------------------

A vector/matrix function takes one or more scalars, vectors, or matrices as arguments
and returns a vector or matrix.

CVXPY is conservative when it determines the sign of an Expression returned by one of these functions.
If any argument to one of these functions
has unknown sign, then the returned Expression will also have unknown sign.
If all arguments have known sign but CVXPY can determine that the returned Expression
would have different signs in different entries (for example, when stacking a positive
Expression and a negative Expression) then the returned Expression will have unknown sign.

.. list-table::
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Curvature |_|
     - Monotonicity

   * - :ref:`bmat([[X11,...,X1q],
       ...,
       [Xp1,...,Xpq]]) <bmat>`

     - :math:`\left[\begin{matrix} X^{(1,1)} &  \cdots &  X^{(1,q)} \\ \vdots &   & \vdots \\ X^{(p,1)} & \cdots &   X^{(p,q)} \end{matrix}\right]`
     - :math:`X^{(i,j)} \in\mathbf{R}^{m_i \times n_j}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`convolve(c, x) <convolve>`

       :math:`c\in\mathbf{R}^m`
     - :math:`c*x`
     - :math:`x\in \mathbf{R}^n`
     - |affine| affine
     - depends |_| on |_| c

   * - :ref:`cumsum(X, axis=0) <cumsum>`

     - cumulative sum along given axis.
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`diag(x) <diag>`

     - :math:`\left[\begin{matrix}x_1  & &  \\& \ddots & \\& & x_n\end{matrix}\right]`
     - :math:`x \in\mathbf{R}^{n}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`diag(X) <diag>`
     - :math:`\left[\begin{matrix}X_{11}  \\\vdots \\X_{nn}\end{matrix}\right]`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`diff(X, k=1, axis=0) <diff>`

       :math:`k \in 0,1,2,\ldots`
     - kth order differences along given axis
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`hstack([X1, ..., Xk]) <hstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \cdots    X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n_i}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`kron(X, Y) <kron>`

       constant :math:`X\in\mathbf{R}^{p \times q}`
     - :math:`\left[\begin{matrix}X_{11}Y & \cdots & X_{1q}Y \\ \vdots  &        & \vdots \\ X_{p1}Y &  \cdots      & X_{pq}Y     \end{matrix}\right]`
     - :math:`Y \in \mathbf{R}^{m \times n}`
     - |affine| affine
     - depends on :math:`X`

   * - :ref:`kron(X, Y) <kron>`

       constant :math:`Y\in\mathbf{R}^{m \times n}`
     - :math:`\left[\begin{matrix}X_{11}Y & \cdots & X_{1q}Y \\ \vdots  &        & \vdots \\ X_{p1}Y &  \cdots      & X_{pq}Y     \end{matrix}\right]`
     - :math:`X \in \mathbf{R}^{p \times q}`
     - |affine| affine
     - depends on :math:`Y`
     
   * - :ref:`outer(x, y) <outer>`

       constant :math:`y \in \mathbf{R}^m`
     - :math:`x y^T`
     - :math:`x \in \mathbf{R}^n`
     - |affine| affine
     - depends on :math:`y`

   * - :ref:`partial_trace(X, dims, axis=0) <ptrace>`

     - partial trace
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`partial_transpose(X, dims, axis=0) <ptrans>`

     - partial transpose
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`reshape(X, (m', n'), order='F') <reshape>`

     - :math:`X' \in\mathbf{R}^{m' \times n'}`
     - :math:`X \in\mathbf{R}^{m \times n}`

       :math:`m'n' = mn`
     - |affine| affine
     - |incr| incr.

   * - :ref:`upper_tri(X) <upper_tri>`

     - flatten the strictly upper-triangular part of :math:`X`
     - :math:`X \in \mathbf{R}^{n \times n}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`vec(X) <vec>`

     - :math:`x' \in\mathbf{R}^{mn}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`vec_to_upper_tri(X, strict=False) <vec_to_upper_tri>`

     - :math:`x' \in\mathbf{R}^{n(n-1)/2}` for ``strict=True``

       :math:`x' \in\mathbf{R}^{n(n+1)/2}` for ``strict=False``
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |affine| affine
     - |incr| incr.

   * - :ref:`vstack([X1, ..., Xk]) <vstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \\ \vdots  \\X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m_i \times n}`
     - |affine| affine
     - |incr| incr.

Clarifications on vector and matrix functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The input to :math:`\texttt{bmat}` is a list of lists of CVXPY expressions.
It constructs a block matrix.
The elements of each inner list are stacked horizontally and then the resulting block matrices are stacked vertically.

The output :math:`y = \mathbf{convolve}(c, x)` has size :math:`n+m-1` and is defined as
:math:`y_k =\sum_{j=0}^{k} c[j]x[k-j]`.

The output :math:`y = \mathbf{vec}(X)` is the matrix :math:`X` flattened in column-major order into a vector.
Formally, :math:`y_i = X_{i \bmod{m}, \left \lfloor{i/m}\right \rfloor }`.

The output :math:`Y = \mathbf{reshape}(X, (m', n'), \text{order='F'})` is the matrix :math:`X` cast into an :math:`m' \times n'` matrix.
The entries are taken from :math:`X` in column-major order and stored in :math:`Y` in column-major order.
Formally, :math:`Y_{ij} = \mathbf{vec}(X)_{m'j + i}`.
If order='C' then :math:`X` will be read in row-major order and :math:`Y` will be written to in row-major order.

The output :math:`y = \mathbf{upper\_tri}(X)` is formed by concatenating partial rows of :math:`X`.
I.e., :math:`y = (X[0,1{:}],\, X[1, 2{:}],\, \ldots, X[n-1, n])`.

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
