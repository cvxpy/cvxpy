.. _advanced:

Advanced Features
=================

This section of the tutorial covers features of CVXPY intended for users with advanced knowledge of convex optimization. We recommend `Convex Optimization <http://www.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe as a reference for any terms you are unfamiliar with.

Dual variables
--------------

You can use CVXPY to find the optimal dual variables for a problem. When you call ``prob.solve()`` each dual variable in the solution is stored in the ``dual_value`` field of the constraint it corresponds to.


.. code:: python

    import cvxpy as cp

    # Create two scalar optimization variables.
    x = cp.Variable()
    y = cp.Variable()

    # Create two constraints.
    constraints = [x + y == 1,
                   x - y >= 1]

    # Form objective.
    obj = cp.Minimize((x - y)**2)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()

    # The optimal dual variable (Lagrange multiplier) for
    # a constraint is stored in constraint.dual_value.
    print("optimal (x + y == 1) dual variable", constraints[0].dual_value)
    print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
    print("x - y value:", (x - y).value)

::

    optimal (x + y == 1) dual variable 6.47610300459e-18
    optimal (x - y >= 1) dual variable 2.00025244976
    x - y value: 0.999999986374

The dual variable for ``x - y >= 1`` is 2. By complementarity this implies that ``x - y`` is 1, which we can see is true. The fact that the dual variable is non-zero also tells us that if we tighten ``x - y >= 1``, (i.e., increase the right-hand side), the optimal value of the problem will increase.

.. _attributes:

Attributes
----------

Variables and parameters can be created with attributes specifying additional properties.
For example, ``Variable(nonneg=True)`` is a scalar variable constrained to be nonnegative.
Similarly, ``Parameter(nonpos=True)`` is a scalar parameter constrained to be nonpositive.
The full constructor for :py:class:`Leaf <cvxpy.expressions.leaf.Leaf>` (the parent class
of :py:class:`Variable <cvxpy.expressions.variable.Variable>` and
:py:class:`Parameter <cvxpy.expressions.constants.parameter.Parameter>`) is given below.

.. function:: Leaf(shape=None, name=None, value=None, nonneg=False, nonpos=False, symmetric=False, diag=False, PSD=False, NSD=False, boolean=False, integer=False)

    Creates a Leaf object (e.g., Variable or Parameter).
    Only one attribute can be active (set to True).

    :param shape: The variable dimensions (0D by default). Cannot be more than 2D.
    :type shape: tuple or int
    :param name: The variable name.
    :type name: str
    :param value: A value to assign to the variable.
    :type value: numeric type
    :param nonneg: Is the variable constrained to be nonnegative?
    :type nonneg: bool
    :param nonpos: Is the variable constrained to be nonpositive?
    :type nonpos: bool
    :param symmetric: Is the variable constrained to be symmetric?
    :type symmetric: bool
    :param hermitian: Is the variable constrained to be Hermitian?
    :type hermitian: bool
    :param diag: Is the variable constrained to be diagonal?
    :type diag: bool
    :param complex: Is the variable complex valued?
    :type complex: bool
    :param imag: Is the variable purely imaginary?
    :type imag: bool
    :param PSD: Is the variable constrained to be symmetric positive semidefinite?
    :type PSD: bool
    :param NSD: Is the variable constrained to be symmetric negative semidefinite?
    :type NSD: bool
    :param boolean:
        Is the variable boolean (i.e., 0 or 1)? True, which constrains
        the entire variable to be boolean, False, or a list of
        indices which should be constrained as boolean, where each
        index is a tuple of length exactly equal to the
        length of shape.
    :type boolean: bool or list of tuple
    :param integer: Is the variable integer? The semantics are the same as the boolean argument.
    :type integer: bool or list of tuple

The ``value`` field of Variables and Parameters can be assigned a value after construction,
but the assigned value must satisfy the object attributes.
A Euclidean projection onto the set defined by the attributes is given by the
:py:meth:`project <cvxpy.expressions.leaf.Leaf.project>` method.

.. code:: python

    p = Parameter(nonneg=True)
    try:
        p.value = -1
    except Exception as e:
        print(e)

    print("Projection:", p.project(-1))

::

    Parameter value must be nonnegative.
    Projection: 0.0

A sensible idiom for assigning values to leaves is
:py:meth:`leaf.value = leaf.project(val) <cvxpy.expressions.leaf.Leaf.project>`,
ensuring that the assigned value satisfies the leaf's properties.
A slightly more efficient variant is
:py:meth:`leaf.project_and_assign(val) <cvxpy.expressions.leaf.Leaf.project_and_assign>`,
which projects and assigns the value directly, without additionally checking
that the value satisfies the leaf's properties.  In most cases ``project`` and
checking that a value satisfies a leaf's properties are cheap operations (i.e.,
:math:`O(n)`), but for symmetric positive semidefinite or negative semidefinite
leaves, the operations compute an eigenvalue decomposition.

Many attributes, such as nonnegativity and symmetry, can be easily specified with constraints.
What is the advantage then of specifying attributes in a variable?
The main benefit is that specifying attributes enables more fine-grained DCP analysis.
For example, creating a variable ``x`` via ``x = Variable(nonpos=True)`` informs the DCP analyzer that ``x`` is nonpositive.
Creating the variable ``x`` via ``x = Variable()`` and adding the constraint ``x >= 0`` separately does not provide any information
about the sign of ``x`` to the DCP analyzer.

.. _semidefinite:

Semidefinite matrices
----------------------

Many convex optimization problems involve constraining matrices to be positive or negative semidefinite (e.g., SDPs).
You can do this in CVXPY in two ways.
The first way is to use
``Variable((n, n), PSD=True)`` to create an ``n`` by ``n`` variable constrained to be symmetric and positive semidefinite. For example,

.. code:: python

    # Creates a 100 by 100 positive semidefinite variable.
    X = cp.Variable((100, 100), PSD=True)

    # You can use X anywhere you would use
    # a normal CVXPY variable.
    obj = cp.Minimize(cp.norm(X) + cp.sum(X))

The second way is to create a positive semidefinite cone constraint using the ``>>`` or ``<<`` operator.
If ``X`` and ``Y`` are ``n`` by ``n`` variables,
the constraint ``X >> Y`` means that :math:`z^T(X - Y)z \geq 0`, for all :math:`z \in \mathcal{R}^n`.
In other words, :math:`(X - Y) + (X - Y)^T` is positive semidefinite.
The constraint does not require that ``X`` and ``Y`` be symmetric.
Both sides of a postive semidefinite cone constraint must be square matrices and affine.

The following code shows how to constrain matrix expressions to be positive or negative
semidefinite (but not necessarily symmetric).

.. code:: python

    # expr1 must be positive semidefinite.
    constr1 = (expr1 >> 0)

    # expr2 must be negative semidefinite.
    constr2 = (expr2 << 0)

To constrain a matrix expression to be symmetric, simply write

.. code:: python

    # expr must be symmetric.
    constr = (expr == expr.T)

You can also use ``Variable((n, n), symmetric=True)`` to create an ``n`` by ``n`` variable constrained to be symmetric.
The difference between specifying that a variable is symmetric via attributes and adding the constraint ``X == X.T`` is that
attributes are parsed for DCP information and a symmetric variable is defined over the (lower dimensional) vector space of symmetric matrices.

.. _mip:

Mixed-integer programs
----------------------

In mixed-integer programs, certain variables are constrained to be boolean (i.e., 0 or 1) or integer valued.
You can construct mixed-integer programs by creating variables with the attribute that they have only boolean or integer valued entries:

.. code:: python

    # Creates a 10-vector constrained to have boolean valued entries.
    x = cp.Variable(10, boolean=True)

    # expr1 must be boolean valued.
    constr1 = (expr1 == x)

    # Creates a 5 by 7 matrix constrained to have integer valued entries.
    Z = cp.Variable((5, 7), integer=True)

    # expr2 must be integer valued.
    constr2 = (expr2 == Z)


Complex valued expressions
--------------------------

By default variables and parameters are real valued.
Complex valued variables and parameters can be created by setting the attribute ``complex=True``.
Similarly, purely imaginary variables and parameters can be created by setting the attributes ``imag=True``.
Expressions containing complex variables, parameters, or constants may be complex valued.
The functions ``is_real``, ``is_complex``, and ``is_imag`` return whether an expression is purely real, complex, or purely imaginary, respectively.

.. code:: python

   # A complex valued variable.
   x = cp.Variable(complex=True)
   # A purely imaginary parameter.
   p = cp.Parameter(imag=True)

   print("p.is_imag() = ", p.is_imag())
   print("(x + 2).is_real() = ", (x + 2).is_real())

::

   p.is_imag() = True
   (x + 2).is_real() = False

The top-level expressions in the problem objective and inequality constraints must be real valued,
but subexpressions may be complex.
Arithmetic and all linear atoms are defined for complex expressions.
The nonlinear atoms ``abs`` and all norms except ``norm(X, p)`` for ``p < 1`` are also defined for complex expressions.
All atoms whose domain is symmetric matrices are defined for Hermitian matrices.
Similarly, the atoms ``quad_form(x, P)`` and ``matrix_frac(x, P)`` are defined for complex ``x`` and Hermitian ``P``.
Lastly, equality and postive semidefinite constraints are defined for complex expressions.

The following additional atoms are provided for working with complex expressions:

* ``real(expr)`` gives the real part of ``expr``.
* ``imag(expr)`` gives the imaginary part of ``expr`` (i.e., ``expr = real(expr) + 1j*imag(expr)``).
* ``conj(expr)`` gives the complex conjugate of ``expr``.
* ``expr.H`` gives the Hermitian (conjugate) transpose of ``expr``.

Transforms
----------

Transforms provide additional ways of manipulating CVXPY objects
beyond the atomic functions.  For example, the :py:class:`indicator
<cvxpy.transforms.indicator>` transform converts a list of constraints into an
expression representing the convex function that takes value 0 when the
constraints hold and :math:`\infty` when they are violated.


.. code:: python

   x = cp.Variable()
   constraints = [0 <= x, x <= 1]
   expr = cp.indicator(constraints)
   x.value = .5
   print("expr.value = ", expr.value)
   x.value = 2
   print("expr.value = ", expr.value)

::

   expr.value = 0.0
   expr.value = inf

The full set of transforms available is discussed in :ref:`transforms-api`.

Problem arithmetic
------------------

For convenience, arithmetic operations have been overloaded for
problems and objectives.
Problem arithmetic is useful because it allows you to write a problem as a
sum of smaller problems.
The rules for adding, subtracting, and multiplying objectives are given below.

.. code:: python

    # Addition and subtraction.

    Minimize(expr1) + Minimize(expr2) == Minimize(expr1 + expr2)

    Maximize(expr1) + Maximize(expr2) == Maximize(expr1 + expr2)

    Minimize(expr1) + Maximize(expr2) # Not allowed.

    Minimize(expr1) - Maximize(expr2) == Minimize(expr1 - expr2)

    # Multiplication (alpha is a positive scalar).

    alpha*Minimize(expr) == Minimize(alpha*expr)

    alpha*Maximize(expr) == Maximize(alpha*expr)

    -alpha*Minimize(expr) == Maximize(-alpha*expr)

    -alpha*Maximize(expr) == Minimize(-alpha*expr)

The rules for adding and multiplying problems are equally straightforward:

.. code:: python

    # Addition and subtraction.

    prob1 + prob2 == Problem(prob1.objective + prob2.objective,
                             prob1.constraints + prob2.constraints)

    prob1 - prob2 == Problem(prob1.objective - prob2.objective,
                             prob1.constraints + prob2.constraints)

    # Multiplication (alpha is any scalar).

    alpha*prob == Problem(alpha*prob.objective, prob.constraints)

Note that the ``+`` operator concatenates lists of constraints,
since this is the default behavior for Python lists.
The in-place operators ``+=``, ``-=``, and ``*=`` are also supported for
objectives and problems and follow the same rules as above.

.. Given the optimization problems :math:`p_1,\ldots,p_n` where each
.. :math:`p_i` is of the form

.. :math:`\begin{array}{ll}
.. \mbox{minimize}  &f_i(x) \\
.. \mbox{subject to} &x \in \mathcal C_i
.. \end{array}`

.. the weighted sum `\sum_{i=1}^n \alpha_i p_i` is the problem

.. :math:`\begin{array}{ll}
.. \mbox{minimize}  &\sum_{i=1}^n \alpha_i f_i(x) \\
.. \mbox{subject to} &x \in \cap_{i=1}^n \mathcal C_i
.. \end{array}`

Solve method options
--------------------

The ``solve`` method takes optional arguments that let you change how CVXPY
solves the problem.

.. function:: solve(solver=None, verbose=False, gp=False, **kwargs)

   Solves a DCP compliant optimization problem.

   :param solver: The solver to use.
   :type solver: str, optional
   :param verbose:  Overrides the default of hiding solver output.
   :type verbose: bool, optional
   :param gp:  If True, parses the problem as a disciplined geometric program instead of a disciplined convex program.
   :type gp: bool, optional
   :param kwargs: Additional keyword arguments specifying solver specific options.
   :return: The optimal value for the problem, or a string indicating why the problem could not be solved.

We will discuss the optional arguments in detail below.

.. _solvers:

Choosing a solver
^^^^^^^^^^^^^^^^^

CVXPY is distributed with the open source solvers `ECOS`_, `ECOS_BB`_, `OSQP`_, and `SCS`_.
Many other solvers can be called by CVXPY if installed separately.
The table below shows the types of problems the supported solvers can handle.

+--------------+----+----+------+-----+-----+-----+
|              | LP | QP | SOCP | SDP | EXP | MIP |
+==============+====+====+======+=====+=====+=====+
| `CBC`_       | X  |    |      |     |     | X   |
+--------------+----+----+------+-----+-----+-----+
| `GLPK`_      | X  |    |      |     |     |     |
+--------------+----+----+------+-----+-----+-----+
| `GLPK_MI`_   | X  |    |      |     |     | X   |
+--------------+----+----+------+-----+-----+-----+
| `OSQP`_      | X  | X  |      |     |     |     |
+--------------+----+----+------+-----+-----+-----+
| `CPLEX`_     | X  | X  | X    |     |     | X   |
+--------------+----+----+------+-----+-----+-----+
| `Elemental`_ | X  | X  | X    |     |     |     |
+--------------+----+----+------+-----+-----+-----+
| `ECOS`_      | X  | X  | X    |     | X   |     |
+--------------+----+----+------+-----+-----+-----+
| `ECOS_BB`_   | X  | X  | X    |     | X   | X   |
+--------------+----+----+------+-----+-----+-----+
| `GUROBI`_    | X  | X  | X    |     |     | X   |
+--------------+----+----+------+-----+-----+-----+
| `MOSEK`_     | X  | X  | X    | X   |     |     |
+--------------+----+----+------+-----+-----+-----+
| `CVXOPT`_    | X  | X  | X    | X   |     |     |
+--------------+----+----+------+-----+-----+-----+
| `SCS`_       | X  | X  | X    | X   | X   |     |
+--------------+----+----+------+-----+-----+-----+


Here EXP refers to problems with exponential cone constraints. The exponential cone is defined as

    :math:`\{(x,y,z) \mid y > 0, y\exp(x/y) \leq z \} \cup \{ (x,y,z) \mid x \leq 0, y = 0, z \geq 0\}`.

You cannot specify cone constraints explicitly in CVXPY, but cone constraints are added when CVXPY converts the problem into standard form.

By default CVXPY calls the solver most specialized to the problem type. For example, `ECOS`_ is called for SOCPs. `SCS`_ can both handle all problems (except mixed-integer programs). `ECOS_BB`_ is called for mixed-integer LPs and SOCPs. If the problem is a QP, CVXPY will use `OSQP`_.

You can change the solver called by CVXPY using the ``solver`` keyword argument. If the solver you choose cannot solve the problem, CVXPY will raise an exception. Here's example code solving the same problem with different solvers.

.. code:: python

    # Solving a problem with different solvers.
    x = cp.Variable(2)
    obj = cp.Minimize(x[0] + cp.norm(x, 1))
    constraints = [x >= 2]
    prob = cp.Problem(obj, constraints)

    # Solve with OSQP.
    prob.solve(solver=cp.OSQP)
    print("optimal value with OSQP:", prob.value)

    # Solve with ECOS.
    prob.solve(solver=cp.ECOS)
    print("optimal value with ECOS:", prob.value)

    # Solve with ECOS_BB.
    prob.solve(solver=cp.ECOS_BB)
    print("optimal value with ECOS_BB:", prob.value)

    # Solve with CVXOPT.
    prob.solve(solver=cp.CVXOPT)
    print("optimal value with CVXOPT:", prob.value)

    # Solve with SCS.
    prob.solve(solver=cp.SCS)
    print("optimal value with SCS:", prob.value)

    # Solve with GLPK.
    prob.solve(solver=cp.GLPK)
    print("optimal value with GLPK:", prob.value)

    # Solve with GLPK_MI.
    prob.solve(solver=cp.GLPK_MI)
    print("optimal value with GLPK_MI:", prob.value)

    # Solve with GUROBI.
    prob.solve(solver=cp.GUROBI)
    print("optimal value with GUROBI:", prob.value)

    # Solve with MOSEK.
    prob.solve(solver=cp.MOSEK)
    print("optimal value with MOSEK:", prob.value)

    # Solve with Elemental.
    prob.solve(solver=cp.ELEMENTAL)
    print("optimal value with Elemental:", prob.value)

    # Solve with CBC.
    prob.solve(solver=cp.CBC)
    print("optimal value with CBC:", prob.value)

    # Solve with CPLEX.
    prob.solve(solver=cp.CPLEX)
    print "optimal value with CPLEX:", prob.value
::

    optimal value with OSQP: 6.0
    optimal value with ECOS: 5.99999999551
    optimal value with ECOS_BB: 5.99999999551
    optimal value with CVXOPT: 6.00000000512
    optimal value with SCS: 6.00046055789
    optimal value with GLPK: 6.0
    optimal value with GLPK_MI: 6.0
    optimal value with GUROBI: 6.0
    optimal value with MOSEK: 6.0
    optimal value with Elemental: 6.0000044085242727
    optimal value with CBC: 6.0
    optimal value with CPLEX: 6.0

Use the ``installed_solvers`` utility function to get a list of the solvers your installation of CVXPY supports.

.. code:: python

    print installed_solvers()

::

    ['CBC', 'CVXOPT', 'MOSEK', 'GLPK', 'GLPK_MI', 'ECOS_BB', 'ECOS', 'SCS', 'GUROBI', 'ELEMENTAL', 'OSQP', 'CPLEX']

Viewing solver output
^^^^^^^^^^^^^^^^^^^^^

All the solvers can print out information about their progress while solving the problem. This information can be useful in debugging a solver error. To see the output from the solvers, set ``verbose=True`` in the solve method.

.. code:: python

    # Solve with ECOS and display output.
    prob.solve(solver=cp.ECOS, verbose=True)
    print "optimal value with ECOS:", prob.value

::

    ECOS 1.0.3 - (c) A. Domahidi, Automatic Control Laboratory, ETH Zurich, 2012-2014.

    It     pcost         dcost      gap     pres    dres     k/t     mu      step     IR
     0   +0.000e+00   +4.000e+00   +2e+01   2e+00   1e+00   1e+00   3e+00    N/A     1 1 -
     1   +6.451e+00   +8.125e+00   +5e+00   7e-01   5e-01   7e-01   7e-01   0.7857   1 1 1
     2   +6.788e+00   +6.839e+00   +9e-02   1e-02   8e-03   3e-02   2e-02   0.9829   1 1 1
     3   +6.828e+00   +6.829e+00   +1e-03   1e-04   8e-05   3e-04   2e-04   0.9899   1 1 1
     4   +6.828e+00   +6.828e+00   +1e-05   1e-06   8e-07   3e-06   2e-06   0.9899   2 1 1
     5   +6.828e+00   +6.828e+00   +1e-07   1e-08   8e-09   4e-08   2e-08   0.9899   2 1 1

    OPTIMAL (within feastol=1.3e-08, reltol=1.5e-08, abstol=1.0e-07).
    Runtime: 0.000121 seconds.

    optimal value with ECOS: 6.82842708233

Solving disciplined geometric programs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the ``solve`` method is called with `gp=True`, the problem is parsed
as a disciplined geometric program instead of a disciplined convex program.
For more information, see the `DGP tutorial </tutorial/dgp/index>`.

Solver stats
------------

When the ``solve`` method is called on a problem object and a solver is invoked,
the problem object records the optimal value, the values of the primal and dual variables,
and several solver statistics.
We have already discussed how to view the optimal value and variable values.
The solver statistics are accessed via the ``problem.solver_stats`` attribute,
which returns a :class:`~cvxpy.problems.problem.SolverStats` object.
For example, ``problem.solver_stats.solve_time`` gives the time it took the solver to solve the problem.

Warm start
----------

When solving the same problem for multiple values of a parameter, many solvers can exploit work from previous solves (i.e., warm start).
For example, the solver might use the previous solution as an initial point or reuse cached matrix factorizations.
Warm start is enabled by default and controlled with the ``warm_start`` solver option.
The code below shows how warm start can accelerate solving a sequence of related least-squares problems.

.. code:: python

    import cvxpy as cp
    import numpy

    # Problem data.
    m = 2000
    n = 1000
    numpy.random.seed(1)
    A = numpy.random.randn(m, n)
    b = cp.Parameter(m)

    # Construct the problem.
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A*x - b)),
                       [x >= 0])

    b.value = numpy.random.randn(m)
    prob.solve()
    print("First solve time:", prob.solve_time)

    b.value = numpy.random.randn(m)
    prob.solve(warm_start=True)
    print("Second solve time:", prob.solve_time)

::

   First solve time: 11.14
   Second solve time: 2.95

The speed up in this case comes from caching the KKT matrix factorization.
If ``A`` were a parameter, factorization caching would not be possible and the benefit of
warm start would only be a good initial point.

Setting solver options
^^^^^^^^^^^^^^^^^^^^^^

The `OSQP`_, `ECOS`_, `ECOS_BB`_, `MOSEK`_, `CBC`_, `CVXOPT`_, and `SCS`_ Python interfaces allow you to set solver options such as the maximum number of iterations. You can pass these options along through CVXPY as keyword arguments.

For example, here we tell SCS to use an indirect method for solving linear equations rather than a direct method.

.. code:: python

    # Solve with SCS, use sparse-indirect method.
    prob.solve(solver=cp.SCS, verbose=True, use_indirect=True)
    print "optimal value with SCS:", prob.value

::

    ----------------------------------------------------------------------------
        SCS v1.0.5 - Splitting Conic Solver
        (c) Brendan O'Donoghue, Stanford University, 2012
    ----------------------------------------------------------------------------
    Lin-sys: sparse-indirect, nnz in A = 13, CG tol ~ 1/iter^(2.00)
    EPS = 1.00e-03, ALPHA = 1.80, MAX_ITERS = 2500, NORMALIZE = 1, SCALE = 5.00
    Variables n = 5, constraints m = 9
    Cones:  linear vars: 6
        soc vars: 3, soc blks: 1
    Setup time: 2.78e-04s
    ----------------------------------------------------------------------------
     Iter | pri res | dua res | rel gap | pri obj | dua obj | kap/tau | time (s)
    ----------------------------------------------------------------------------
         0| 4.60e+00  5.78e-01       nan      -inf       inf       inf  3.86e-05
        60| 3.92e-05  1.12e-04  6.64e-06  6.83e+00  6.83e+00  1.41e-17  9.51e-05
    ----------------------------------------------------------------------------
    Status: Solved
    Timing: Total solve time: 9.76e-05s
        Lin-sys: avg # CG iterations: 1.00, avg solve time: 2.24e-07s
        Cones: avg projection time: 4.90e-08s
    ----------------------------------------------------------------------------
    Error metrics:
    |Ax + s - b|_2 / (1 + |b|_2) = 3.9223e-05
    |A'y + c|_2 / (1 + |c|_2) = 1.1168e-04
    |c'x + b'y| / (1 + |c'x| + |b'y|) = 6.6446e-06
    dist(s, K) = 0, dist(y, K*) = 0, s'y = 0
    ----------------------------------------------------------------------------
    c'x = 6.8284, -b'y = 6.8285
    ============================================================================
    optimal value with SCS: 6.82837896975

Here is the complete list of solver options.

`OSQP`_ options:

``'max_iter'``
    maximum number of iterations (default: 10,000).

``'eps_abs'``
    absolute accuracy (default: 1e-4).

``'eps_rel'``
    relative accuracy (default: 1e-4).

For others see `OSQP documentation <http://osqp.org/docs/interfaces/solver_settings.html>`_.

`ECOS`_ options:

``'max_iters'``
    maximum number of iterations (default: 100).

``'abstol'``
    absolute accuracy (default: 1e-7).

``'reltol'``
    relative accuracy (default: 1e-6).

``'feastol'``
    tolerance for feasibility conditions (default: 1e-7).

``'abstol_inacc'``
    absolute accuracy for inaccurate solution (default: 5e-5).

``'reltol_inacc'``
    relative accuracy for inaccurate solution (default: 5e-5).

``'feastol_inacc'``
    tolerance for feasibility condition for inaccurate solution (default: 1e-4).

`ECOS_BB`_ options:

``'mi_max_iters'``
    maximum number of branch and bound iterations (default: 1000)

``'mi_abs_eps'``
    absolute tolerance between upper and lower bounds (default: 1e-6)

``'mi_rel_eps'``
    relative tolerance, (U-L)/L, between upper and lower bounds (default: 1e-3)

`MOSEK`_ options:

``'mosek_params'``
    A dictionary of MOSEK parameters. Refer to MOSEK's Python or C API for
    details. Note that if parameters are given as string-value pairs, parameter
    names must be of the form ``'MSK_DPAR_BASIS_TOL_X'`` as in the C API.
    Alternatively, Python enum options like ``'mosek.dparam.basis_tol_x'`` are
    also supported.

``'save_file'``
    The name of a file where MOSEK will save the problem just before optimization.
    Refer to MOSEK documentation for a list of supported file formats. File format
    is chosen based on the extension.

`CVXOPT`_ options:

``'max_iters'``
    maximum number of iterations (default: 100).

``'abstol'``
    absolute accuracy (default: 1e-7).

``'reltol'``
    relative accuracy (default: 1e-6).

``'feastol'``
    tolerance for feasibility conditions (default: 1e-7).

``'refinement'``
    number of iterative refinement steps after solving KKT system (default: 1).

``'kktsolver'``
    The KKT solver used. The default, "chol", does a Cholesky factorization with preprocessing to make A and [A; G] full rank.
    The "robust" solver does an LDL factorization without preprocessing.
    It is slower, but more robust.

`SCS`_ options:

``'max_iters'``
    maximum number of iterations (default: 2500).

``'eps'``
    convergence tolerance (default: 1e-4).

``'alpha'``
    relaxation parameter (default: 1.8).

``'scale'``
    balance between minimizing primal and dual residual (default: 5.0).

``'normalize'``
    whether to precondition data matrices (default: True).

``'use_indirect'``
    whether to use indirect solver for KKT sytem (instead of direct) (default: True).

`CBC`_ options:

Cut-generation through `CGL`_

General remarks:
    - some of these cut-generators seem to be buggy (observed problems with AllDifferentCuts, RedSplitCuts, LandPCuts, PreProcessCuts)
    - a few of these cut-generators will generate noisy output even if ``'verbose=False'``

The following cut-generators are available:
    ``GomoryCuts``, ``MIRCuts``, ``MIRCuts2``, ``TwoMIRCuts``, ``ResidualCapacityCuts``, ``KnapsackCuts`` ``FlowCoverCuts``, ``CliqueCuts``, ``LiftProjectCuts``, ``AllDifferentCuts``, ``OddHoleCuts``, ``RedSplitCuts``, ``LandPCuts``, ``PreProcessCuts``, ``ProbingCuts``, ``SimpleRoundingCuts``.

``'CutGenName'``
    if cut-generator is activated (e.g. ``'GomoryCuts=True'``)

`CPLEX`_ options:

``'cplex_params'``
    a dictionary where the key-value pairs are composed of parameter names (as used in the CPLEX Python API) and parameter values. For example, to set the advance start switch parameter (i.e., CPX_PARAM_ADVIND), use "advance" for the parameter name. For the data consistency checking and modeling assistance parameter (i.e., CPX_PARAM_DATACHECK), use "read.datacheck" for the parameter name, and so on.

``'cplex_filename'``
    a string specifying the filename to which the problem will be written. For example, use "model.lp", "model.sav", or "model.mps" to export to the LP, SAV, and MPS formats, respectively.

Getting the standard form
-------------------------

If you are interested in getting the standard form that CVXPY produces for a
problem, you can use the ``get_problem_data`` method. When a problem is solved, 
a :class:`~cvxpy.reductions.solvers.solving_chain.SolvingChain` passes a
low-level representation that is compatible with the targeted solver to a
solver, which solves the problem. This method returns that low-level
representation, along with a ``SolvingChain`` and metadata for unpacking
a solution into the problem. This low-level representation closely resembles,
but is not identitical to, the
arguments supplied to the solver.

A solution to the equivalent low-level problem can be obtained via the
data by invoking the ``solve_via_data`` method of the returned solving
chain, a thin wrapper around the code external to CVXPY that further
processes and solves the problem. Invoke the ``unpack_results`` method
to recover a solution to the original problem.

For example:

.. code:: python

  problem = cp.Problem(objective, constraints)
  data, chain, inverse_data = problem.get_problem_data(cp.SCS)
  # calls SCS using `data`
  soln = chain.solve_via_data(problem, data)
  # unpacks the solution returned by SCS into `problem`
  problem.unpack_results(soln, chain, inverse_data)

Alternatively, the ``data`` dictionary returned by this method
contains enough information to bypass CVXPY and call the solver
directly.

For example:

.. code:: python

  problem = cp.Problem(objective, constraints)
  data, _, _ = problem.get_problem_data(cp.SCS)

  import scs
  probdata = {
    'A': data['A'],
    'b': data['b'],
    'c': data['c'],
  }
  cone_dims = data['dims']
  cones = {
      "f": cone_dims.zero,
      "l": cone_dims.nonpos,
      "q": cone_dims.soc,
      "ep": cone_dims.exp,
      "s": cone_dims.psd,
  }
  soln = scs.solve(data, cones)

The structure of the data dict that CVXPY returns depends on the solver. For
details, print the dictionary, or consult the solver interfaces in
``cvxpy/reductions/solvers``.

Reductions
----------

CVXPY uses a system of **reductions** to rewrite problems from
the form provided by the user into the standard form that a solver will accept.
A reduction is a transformation from one problem to an equivalent problem.
Two problems are equivalent if a solution of one can be converted efficiently
to a solution of the other.
Reductions take a CVXPY Problem as input and output a CVXPY Problem.
The full set of reductions available is discussed in :ref:`reductions-api`.

.. _OSQP: https://osqp.org/
.. _CVXOPT: http://cvxopt.org/
.. _ECOS: https://www.embotech.com/ECOS
.. _ECOS_BB: https://github.com/embotech/ecos#mixed-integer-socps-ecos_bb
.. _SCS: http://github.com/cvxgrp/scs
.. _GLPK: https://www.gnu.org/software/glpk/
.. _GLPK_MI: https://www.gnu.org/software/glpk/
.. _GUROBI: http://www.gurobi.com/
.. _MOSEK: https://www.mosek.com/
.. _Elemental: http://libelemental.org/
.. _CBC: https://projects.coin-or.org/Cbc
.. _CGL: https://projects.coin-or.org/Cgl
.. _CPLEX: https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/
