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

.. list-table::
   :class: scalar-dcp
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Sign
     - Curvature |_|
     - Monotonicity

   * - :ref:`dotsort(X,W) <dotsort>`

       constant :math:`W \in \mathbf{R}^{o \times p}`
     - :math:`\langle sort\left(vec(X)\right), sort\left(vec(W)\right) \rangle`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - depends on :math:`X`, :math:`W`
     - |convex| convex
     - |incr| for :math:`\min(W) \geq 0`

       |decr| for :math:`\max(W) \leq 0`

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
   :class: element-dcp
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Sign
     - Curvature |_|
     - Monotonicity

   * - :ref:`abs(x) <abs>`

     - :math:`\lvert x \rvert`
     - :math:`x \in \mathbf{C}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`

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

   * - :ref:`vec_to_upper_tri(X, strict=False) <vec-to-upper-tri>`

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

.. include:: ../../functions/functions-table.rst
.. raw:: html

    <script type="text/javascript">
        $(document).ready(function() {
            $("table.atomic-functions").hide();
            $("table.atomic-functions").closest('.dataTables_wrapper').hide();
        });
    </script>