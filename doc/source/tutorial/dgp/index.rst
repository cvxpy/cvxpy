.. _dgp:

Disciplined Geometric Programming
=================================

Disciplined geometric programming (DGP) is an analog of DCP for
*log-log convex* functions, that is, functions of positive variables that
are convex with respect to the geometric mean instead of the arithmetic mean.

While DCP is a ruleset for constructing convex programs, DGP
is a ruleset for log-log convex programs (LLCPs), which are problems that are
convex after the variables, objective functions, and constraint functions are
replaced with their logs, an operation that we refer to as a *log-log*
transformation. Every geometric program (GP) and generalized geometric program
(GGP) is an LLCP, but there are LLCPs that are neither GPs nor GGPs.

CVXPY lets you form and solve DGP problems, just as it does for DCP
problems. For example, the following code solves a simple geometric program,

.. code:: python

    import cvxpy as cp

    # DGP requires Variables to be declared positive via `pos=True`.
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)

    objective_fn = x * y * z
    constraints = [
      4 * x * y * z + 2 * x * z <= 10, x <= 2*y, y <= 2*x, z >= 1]
    problem = cp.Problem(cp.Maximize(objective_fn), constraints)
    problem.solve(gp=True)
    print("Optimal value: ", problem.value)    
    print("x: ", x.value)
    print("y: ", y.value)
    print("z: ", z.value)

and it prints the below output.

::

    Optimal value: 1.9999999938309496
    x: 0.9999999989682057
    y: 1.999999974180587
    z: 1.0000000108569758

Note that to solve DGP problems, you must pass the option
``gp=True`` to the ``solve()`` method.

This section explains what DGP is, and it shows how to construct and solve DGP
problems using CVXPY. At the end of the section are tables listing all the
atoms that can be used in DGP problems, similar to the tables presented in
the section on :ref:`DCP atoms <functions>`.

For an in-depth reference on DGP, see our
`accompanying paper <https://web.stanford.edu/~boyd/papers/dgp.html>`_.
For interactive code examples, check out our :ref:`notebooks <dgp-examples>`.

*Note: DGP is a recent addition to CVXPY. If you have feedback, please file an
issue or make a pull request on* `Github <https://github.com/cvxpy/cvxpy>`_.

Log-log curvature
-----------------

Just as every Expression in CVXPY has a curvature (constant, affine, convex, concave,
or unknown), every Expression also has a log-log curvature.

A function :math:`f : D \subseteq \mathbf{R}^n_{++} \to \mathbf{R}` is said to
be log-log convex if the function :math:`F(u) = \log f(e^u)`, with domain
:math:`\{u \in \mathbf{R}^n : e^u \in D\}`, is convex (where
:math:`\mathbf{R}^n_{++}` denotes the set of positive reals and the logarithm
and exponential are meant elementwise); the function :math:`F` is called the
log-log transformation of `f`. The function :math:`f` is log-log concave if
:math:`F` is concave, and it is log-log affine if :math:`F` is affine.

Every log-log affine function has the form

.. math::

    f(x) = cx_1^{a_1}x_2^{a_2} \ldots x_n^{a_n}

where :math:`x` is in :math:`\mathbf{R}^n_{++}`, the :math:`a_i` are
real numbers, and :math:`c` is a positive scalar. In the context of
geometric programming, such a function is called a monomial function.
A sum of monomials, known as a posynomial function in geometric programming, is
a log-log convex function;  A table of all the `atoms with known log-log
curvature <dgp-atoms>`_ is presented at the end of this page.

In the below table, :math:`F` is the log-log transformation of :math:`f`,
:math:`u=\log x`, and :math:`v=\log y`,
where :math:`x` and :math:`y` are in the domain of :math:`f`

=================      =======
Log-Log Curvature      Meaning
=================      =======
log-log constant       :math:`F` is a constant (so `f` is a positive constant)
log-log affine         :math:`F(\theta u + (1-\theta)v) = \theta F(u) + (1-\theta)F(v), \; \forall u, \; v,\; \theta \in [0,1]`
log-log convex         :math:`F(\theta u + (1-\theta)v) \leq \theta F(u) + (1-\theta)F(v), \; \forall u, \; v,\; \theta \in [0,1]`
log-log concave        :math:`F(\theta u + (1-\theta)v) \geq \theta F(u) + (1-\theta)F(v), \; \forall u, \; v,\; \theta \in [0,1]`
unknown                DGP analysis cannot determine the curvature
=================      =======

CVXPY's log-log curvature analysis can flag
Expressions as unknown even when they are log-log convex or log-log concave.
Note that any log-log constant expression is also log-log affine, and any
log-log affine expression is log-log convex and log-log concave.

The log-log curvature of an Expression is stored in its
:code:`.log_log_curvature` attribute. For example, running the following
script

.. code:: python

    import cvxpy as cp

    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)

    constant = cp.Constant(2.0)
    monomial = constant * x * y
    posynomial = monomial + (x ** 1.5) * (y ** -1)
    reciprocal = posynomial ** -1 
    unknown = reciprocal + posynomial

    print(constant.log_log_curvature)
    print(monomial.log_log_curvature)
    print(posynomial.log_log_curvature)
    print(reciprocal.log_log_curvature)
    print(unknown.log_log_curvature)

prints the following output.

::

    LOG-LOG CONSTANT
    LOG-LOG AFFINE
    LOG-LOG CONVEX
    LOG-LOG CONCAVE
    UNKNOWN
  
You can also check the log-log curvature of an Expression by
calling the methods
:code:`is_log_log_constant()`, :code:`is_log_log_affine()`,
:code:`is_log_log_convex()`, :code:`is_log_log_concave()`. For example,
:code:`posynomial.is_log_log_convex()` would evaluate to :code:`True`.

Log-log curvature rules
-----------------------
For an Expression to have known log-log curvature, all of the Constants,
Variables, and Parameters it refers to must be elementwise positive. A
Constant is positive if its numerical value is positive. Variables
and Parameters are positive only if the keyword argument :code:`pos=True`
is supplied to their constructors (e.g.,
:code:`x = cvxpy.Variable(shape=(), pos=True)`). To summarize,
when formulating a DGP problem, *all Constants should be elementwise positive,
and all Variables and Parameters must be constructed with the attribute*
:code:`pos=True`.

DGP analysis is exactly analogous to DCP analysis. It is based on a library
of atoms (functions) with known monotonicity and log-log curvature and a
a single composition rule. The `library of atoms <dgp-atoms>`_ is presented
at the end of this page; the composition rule is stated below.

A function :math:`f(expr_1, expr_2, ..., expr_n)` is log-log convex if :math:`\text{ } f`
is a log-log convex function and for each :math:`expr_{i}` one of the following
conditions holds:

-  :math:`f` is increasing in argument :math:`i` and :math:`expr_{i}` is log-log convex.
-  :math:`f` is decreasing in argument :math:`i` and :math:`expr_{i}` is
   log-log concave.
-  :math:`expr_{i}` is log-log affine.

A function :math:`f(expr_1, expr_2, ..., expr_n)` is log-log concave if :math:`\text{ } f`
is a log-log concave function and for each :math:`expr_{i}` one of the following
conditions holds:

-  :math:`f` is increasing in argument :math:`i` and :math:`expr_{i}` is
   log-log concave.
-  :math:`f` is decreasing in argument :math:`i` and :math:`expr_{i}` is log-log convex.
-  :math:`expr_{i}` is log-log affine.

A function :math:`f(expr_1, expr_2, ..., expr_n)` is log-log affine if :math:`\text{ } f`
is an log-log affine function and each :math:`expr_{i}` is log-log affine.

If none of the three rules apply, the expression :math:`f(expr_1, expr_2, ...,
expr_n)` is marked as having unknown curvature.

If an Expression satisfies the composition rule, we colloquially say that
the Expression "is DGP." You can check whether an Expression is DGP
by calling the method :code:`is_dgp()`. For example, the assertions
in the following code block will pass.

.. code:: python

    import cvxpy as cp

    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)

    monomial = 2.0 * constant * x * y
    posynomial = monomial + (x ** 1.5) * (y ** -1)

    assert monomial.is_dgp()
    assert posynomial.is_dgp()

An Expression is DGP precisely when it has known log-log curvature, which means
at least one of the methods :code:`is_log_log_constant()`,
:code:`is_log_log_affine()`, :code:`is_log_log_convex()`,
:code:`is_log_log_concave()` will return :code:`True`.

DGP problems
------------

A :class:`~cvxpy.problems.problem.Problem` is constructed from an objective and
a list of constraints. If a problem follows the DGP rules, it is guaranteed to
be an LLCP and solvable by CVXPY. The DGP rules require that the problem
objective have one of two forms:

-  Minimize(log-log convex)
-  Maximize(log-log concave)

The only valid constraints under the DGP rules are

-  log-log affine == log-log affine
-  log-log convex <= log-log concave
-  log-log concave >= log-log convex

You can check that a problem, constraint, or objective satisfies the DGP
rules by calling ``object.is_dgp()``. Here are some examples of DGP and
non-DGP problems:

.. code:: python

    import cvxpy as cp

    # DGP requires Variables to be declared positive via `pos=True`.
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)

    objective_fn = x * y * z
    constraints = [
      4 * x * y * z + 2 * x * z <= 10, x <= 2*y, y <= 2*x, z >= 1]
    assert objective_fn.is_log_log_concave()
    assert all(constraint.is_dgp() for constraint in constraints)
    problem = cp.Problem(cp.Maximize(objective_fn), constraints)
    assert problem.is_dgp()

    # All Variables must be declared as positive for an Expression to be DGP.
    w = cp.Variable()
    objective_fn = w * x * y
    assert not objective_fn.is_dgp()
    problem = cp.Problem(cp.Maximize(objective_fn), constraints)
    assert not problem.is_dgp()

CVXPY will raise an exception if you call ``problem.solve(gp=True)`` on a
non-DGP problem.

.. _dgp-atoms:

DGP atoms
---------

This section of the tutorial describes the DGP atom library, that is,
the atomic functions with known log-log curvature and monotonicity.
CVXPY uses the function information in this section and the DGP rules
to mark expressions with a log-log curvature. Note that every DGP expression
is positive.

Infix operators
***************
The infix operators ``+, *, /`` are treated as atoms. The operators
``*`` and ``/`` are log-log affine functions. The operator ``+``
is log-log convex in both its arguments.

Note that in CVXPY, ``expr1 * expr2`` denotes matrix multiplication
when ``expr1`` and ``expr2`` are matrices; if you're running Python 3,
you can alternatively use the ``@`` operator for matrix multiplication.
Regardless of your Python version, you can also use the :ref:`matmul atom
<matmul>` to multiply two matrices. To multiply two arrays or matrices
elementwise, use the :ref:`multiply atom <multiply>`. Finally,
to take the product of the entries of an Expression, use
the :ref:`prod atom <prod>`.

Transpose
*********
The transpose of any expression can be obtained using the syntax
``expr.T``. Transpose is a log-log affine function.

Power
*****
For any CVXPY expression ``expr``, the power operator ``expr**p`` is equivalent
to the function ``power(expr, p)``. Taking powers is a log-log affine function.

Scalar functions
****************

A scalar function takes one or more scalars, vectors, or matrices as arguments
and returns a scalar. Note that several of these atoms may be
applied along an axis; see the API reference or the :ref:`DCP atoms
tutorial <functions>` for more information.

.. |_| unicode:: 0xA0
   :trim:

.. list-table::
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Log-log curvature |_|
     - Monotonicity

   * - :ref:`geo_mean(x) <geo-mean>`

       :ref:`geo_mean(x, p) <geo-mean>`

       :math:`p \in \mathbf{R}^n_{+}`

       :math:`p \neq 0`
     - :math:`x_1^{1/n} \cdots x_n^{1/n}`

       :math:`\left(x_1^{p_1} \cdots x_n^{p_n}\right)^{\frac{1}{\mathbf{1}^T p}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`harmonic_mean(x) <harmonic-mean>`
     - :math:`\frac{n}{\frac{1}{x_1} + \cdots + \frac{1}{x_n}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |concave| log-log concave
     - |incr| incr.

   * - :ref:`max(X) <max>`

     - :math:`\max_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`min(X) <min>`

     - :math:`\min_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |concave| log-log concave
     - |incr| incr.

   * - :ref:`norm(x) <norm>`

       norm(x, 2)

     - :math:`\sqrt{\sum_{i} \lvert x_{i} \rvert^2 }`
     - :math:`X \in\mathbf{R}^{n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`norm(X, "fro") <norm>`
     - :math:`\sqrt{\sum_{ij}X_{ij}^2 }`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`norm(X, 1) <norm>`
     - :math:`\sum_{ij}\lvert X_{ij} \rvert`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`norm(X, "inf") <norm>`
     - :math:`\max_{ij} \{\lvert X_{ij} \rvert\}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p \geq 1`

       or ``p = 'inf'``
     - :math:`\|X\|_p = \left(\sum_{ij} |X_{ij}|^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`0 < p < 1`
     - :math:`\|X\|_p = \left(\sum_{ij} X_{ij}^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`prod(X) <prod>`

     - :math:`\prod_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`quad_form(x, P) <quad-form>`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`, :math:`P \in \mathbf{R}^{n \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`quad_over_lin(X, y) <quad-over-lin>`
     - :math:`\left(\sum_{ij}X_{ij}^2\right)/y`
     - :math:`x \in \mathbf{R}^n_{++}`

       :math:`y > 0`
     - |convex| log-log convex
     - |incr| in :math:`X_{ij}`

       |decr| decr. in :math:`y`

   * - :ref:`sum(X) <sum>`

     - :math:`\sum_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`sum_squares(X) <sum-squares>`

     - :math:`\sum_{ij}X_{ij}^2`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`trace(X) <trace>`

     - :math:`\mathrm{tr}\left(X \right)`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`pf_eigenvalue(X) <pf-eigenvalue>`

     - spectral radius of :math:`X`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     - |convex| log-log convex
     - |incr| incr.

Elementwise functions
*********************

These functions operate on each element of their arguments. For example, if
``X`` is a 5 by 4 matrix variable, then ``sqrt(X)`` is a 5 by 4 matrix
expression. ``sqrt(X)[1, 2]`` is equivalent to ``sqrt(X[1, 2])``.

Elementwise functions that take multiple arguments, such as ``maximum`` and
``multiply``, operate on the corresponding elements of each argument.  For
example, if ``X`` and ``Y`` are both 3 by 3 matrix variables, then ``maximum(X,
Y)`` is a 3 by 3 matrix expression.  ``maximum(X, Y)[2, 0]`` is equivalent to
``maximum(X[2, 0], Y[2, 0])``. This means all arguments must have the same
dimensions or be scalars, which are promoted.

.. list-table::
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Curvature |_|
     - Monotonicity

   * - :ref:`diff_pos(x, y) <diff-pos>`
     - :math:`x - y`
     - :math:`0 < y < x`
     - |concave| log-log concave
     - |incr| incr.  in :math:`x`

       |decr| decr. in :math:`y`

   * - :ref:`entr(x) <entr>`

     - :math:`-x \log (x)`
     - :math:`0 < x < 1`
     - |concave| log-log concave
     - None

   * - :ref:`exp(x) <exp>`

     - :math:`e^x`
     - :math:`x > 0`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`log(x) <log>`

     - :math:`\log(x)`
     - :math:`x > 1`
     - |concave| log-log concave
     - |incr| incr.

   * - :ref:`maximum(x, y) <maximum>`

     - :math:`\max \left\{x, y\right\}`
     - :math:`x,y > 0`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`minimum(x, y) <minimum>`
     - :math:`\min \left\{x, y\right\}`
     - :math:`x, y > 0`
     - |concave| log-log concave
     - |incr| incr.

   * - :ref:`multiply(x, y) <multiply>`
     - :math:`x*y`
     - :math:`x, y > 0`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`one_minus_pos(x) <one-minus-pos>`
     - :math:`1 - x`
     - :math:`0 < x < 1`
     - |concave| log-log concave
     - |decr| decr.

   * - :ref:`power(x, 0) <power>`
     - :math:`1`
     - :math:`x > 0`
     - constant
     - constant

   * - :ref:`power(x, p) <power>`
     - :math:`x`
     - :math:`x > 0`
     - |affine| log-log affine
     - |incr| for :math:`p > 0`

       |decr| for :math:`p < 0`

   * - :ref:`sqrt(x) <sqrt>`


     - :math:`\sqrt x`
     - :math:`x > 0`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`square(x) <square>`

     - :math:`x^2`
     - :math:`x > 0`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`xexp(x) <xexp>`

     - :math:`x e^x`
     - :math:`x > 0`
     - |convex| log-log convex
     - |incr| incr.

Vector/matrix functions
***********************

A vector/matrix function takes one or more scalars, vectors, or matrices as arguments
and returns a vector or matrix.

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
     - :math:`X^{(i,j)} \in\mathbf{R}^{m_i \times n_j}_{++}`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`diag(x) <diag>`

     - :math:`\left[\begin{matrix}x_1  & &  \\& \ddots & \\& & x_n\end{matrix}\right]`
     - :math:`x \in\mathbf{R}^{n}_{++}`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`diag(X) <diag>`
     - :math:`\left[\begin{matrix}X_{11}  \\\vdots \\X_{nn}\end{matrix}\right]`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`eye_minus_inv(X) <eye-minus-inv>`
     - :math:`(I - X)^{-1}`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}, \lambda_{\text{pf}}(X) < 1`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`gmatmul(A, x) <gmatmul>`

       :math:`A \in \mathbf{R}^{m \times n}`
     - :math:`\left[\begin{matrix}\prod_{j=1}^n x_j^{A_{1j}} \\\vdots \\\prod_{j=1}^n x_j^{A_{mj}}\end{matrix}\right]`
     - :math:`x \in \mathbf{R}^n_{++}`
     - |affine| log-log affine
     - |incr| for :math:`A_{ij} \geq 0`

       |decr| for :math:`A_{ij} \leq 0`

   * - :ref:`hstack([X1, ..., Xk]) <hstack>`
     - :math:`\left[\begin{matrix}X^{(1)}  \cdots    X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n_i}_{++}`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`matmul(X, Y) <matmul>`
     - :math:`XY`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}, Y \in\mathbf{R}^{n \times p}_{++}``
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`resolvent(X) <resolvent>`
     - :math:`(sI - X)^{-1}`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}, \lambda_{\text{pf}}(X) < s`
     - |convex| log-log convex
     - |incr| incr.

   * - :ref:`reshape(X, (m', n')) <reshape>`

     - :math:`X' \in\mathbf{R}^{m' \times n'}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`

       :math:`m'n' = mn`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`vec(X) <vec>`

     - :math:`x' \in\mathbf{R}^{mn}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |affine| log-log affine
     - |incr| incr.

   * - :ref:`vstack([X1, ..., Xk]) <vstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \\ \vdots  \\X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m_i \times n}_{++}`
     - |affine| log-log affine
     - |incr| incr.

.. |positive| image:: ../functions/functions_files/positive.svg
              :width: 15px
              :height: 15px

.. |negative| image:: ../functions/functions_files/negative.svg
              :width: 15px
              :height: 15px

.. |unknown| image:: ../functions/functions_files/unknown.svg
              :width: 15px
              :height: 15px

.. |convex| image:: ../functions/functions_files/convex.svg
              :width: 15px
              :height: 15px

.. |concave| image:: ../functions/functions_files/concave.svg
              :width: 15px
              :height: 15px

.. |affine| image:: ../functions/functions_files/affine.svg
              :width: 15px
              :height: 15px

.. |incr| image:: ../functions/functions_files/increasing.svg
              :width: 15px
              :height: 15px

.. |decr| image:: ../functions/functions_files/decreasing.svg
              :width: 15px
              :height: 15px
