.. _solvers:

Solver Features
=================

Solve method options
--------------------

The ``solve`` method takes optional arguments that let you change how CVXPY
parses and solves the problem.

.. function:: solve(solver=None, verbose=False, gp=False, qcp=False, requries_grad=False, enforce_dpp=False, **kwargs)

   Solves the problem using the specified method.

   Populates the :code:`status` and :code:`value` attributes on the
   problem object as a side-effect.

   :param solver: The solver to use.
   :type solver: str, optional
   :param solver_path:  The solvers to use with optional arguments.
            The function tries the solvers in the given order and
            returns the first solver's solution that succeeds.
   :type solver_path: list of (str, dict) tuples or strings, optional   
   :param verbose:  Overrides the default of hiding solver output.
   :type verbose: bool, optional
   :param gp:  If ``True``, parses the problem as a disciplined geometric program instead of a disciplined convex program.
   :type gp: bool, optional
   :param qcp:  If ``True``, parses the problem as a disciplined quasiconvex program instead of a disciplined convex program.
   :type qcp: bool, optional
   :param requires_grad: Makes it possible to compute gradients of a solution
        with respect to Parameters by calling ``problem.backward()`` after
        solving, or to compute perturbations to the variables given perturbations to
        Parameters by calling ``problem.derivative()``.

        Gradients are only supported for DCP and DGP problems, not
        quasiconvex problems. When computing gradients (i.e., when
        this argument is True), the problem must satisfy the DPP rules.
   :type requires_grad: bool, optional
   :param enforce_dpp: When True, a ``DPPError`` will be thrown when trying to solve
        a non-DPP problem (instead of just a warning). Only relevant for
        problems involving Parameters. Defaults to ``False``.
   :type enforce_dpp: bool, optional
   :param ignore_dpp: When True, DPP problems will be treated as non-DPP,
        which may speed up compilation. Defaults to False.
   :type ignore_dpp: bool, optional
   :param kwargs: Additional keyword arguments specifying solver specific options.
   :return: The optimal value for the problem, or a string indicating why the problem could not be solved.

We will discuss the optional arguments in detail below.

.. _solvers:

Choosing a solver
^^^^^^^^^^^^^^^^^

CVXPY is distributed with the open source solvers `CLARABEL`_, `OSQP`_, and `SCS`_.
Many other solvers can be called by CVXPY if installed separately.
The table below shows the types of problems the supported solvers can handle.

+----------------+----+----+------+-----+-----+-----+-----+
|                | LP | QP | SOCP | SDP | EXP | POW | MIP |
+================+====+====+======+=====+=====+=====+=====+
| `CBC`_         | X  |    |      |     |     |     | X   |
+----------------+----+----+------+-----+-----+-----+-----+
| `CLARABEL`_    | X  | X  | X    |  X  |  X  |  X  |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `COPT`_        | X  | X  | X    |  X  |  X  |     | X** |
+----------------+----+----+------+-----+-----+-----+-----+
| `DAQP`_        | X  | X  |      |     |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `GLOP`_        | X  |    |      |     |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `GLPK`_        | X  |    |      |     |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `GLPK_MI`_     | X  |    |      |     |     |     | X   |
+----------------+----+----+------+-----+-----+-----+-----+
| `OSQP`_        | X  | X  |      |     |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `PIQP`_        | X  | X  |      |     |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `PROXQP`_      | X  | X  |      |     |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `PDLP`_        | X  |    |      |     |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `CPLEX`_       | X  | X  | X    |     |     |     | X   |
+----------------+----+----+------+-----+-----+-----+-----+
| `NAG`_         | X  | X  | X    |     |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `ECOS`_        | X  | X  | X    |     | X   |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `GUROBI`_      | X  | X  | X    |     |     |     | X   |
+----------------+----+----+------+-----+-----+-----+-----+
| `MOSEK`_       | X  | X  | X    | X   | X   | X   | X** |
+----------------+----+----+------+-----+-----+-----+-----+
| `CVXOPT`_      | X  | X  | X    | X   |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `SDPA`_ \*\*\* | X  | X  | X    | X   |     |     |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `SCS`_         | X  | X  | X    | X   | X   | X   |     |
+----------------+----+----+------+-----+-----+-----+-----+
| `SCIP`_        | X  | X  | X    |     |     |     | X   |
+----------------+----+----+------+-----+-----+-----+-----+
| `XPRESS`_      | X  | X  | X    |     |     |     | X   |
+----------------+----+----+------+-----+-----+-----+-----+
| `SCIPY`_       | X  |    |      |     |     |     | X*  |
+----------------+----+----+------+-----+-----+-----+-----+

(*) Mixed-integer LP only.

(**) Except mixed-integer SDP. For COPT, mixed-integer EXP also not supported so far.

(\*\*\*) Multiprecision support is available on SDPA if the appropriate SDPA package is installed. With multiprecision support, SDPA can solve your problem with much smaller ``epsilonDash`` and/or ``epsilonStar`` parameters. These parameters must be manually adjusted to achieve the desired degree of precision. Please see the solver website for details. SDPA can also solve some ill-posed problems with multiprecision support.

LP - Linear Programming refers to problems with a linear objective function and linear constraints.

QP - Quadratic Programming refers to problems with a quadratic objective function and linear constraints.

SOCP - Second-Order Cone Programming refers to problems with second-order cone constraints. The second-order cone is defined as

    :math:`\mathcal{C}_{n+1} = \left\{\begin{bmatrix} x \\ t \end{bmatrix} \mid x \in \mathbb{R}^n , t \in \mathbb{R} , \| x \|_2 \leq t\right\}`

SDP - Semidefinite Programming refers to problems with :ref:`semidefinite matrix constraints <Semidefinite matrices>`.

EXP - refers to problems with exponential cone constraints. The exponential cone is defined as

    :math:`\{(x,y,z) \mid y > 0, y\exp(x/y) \leq z \} \cup \{ (x,y,z) \mid x \leq 0, y = 0, z \geq 0\}`.

POW - refers to problems with 3-dimensional power cone constraints. The 3D power cone is defined as

    :math:`\{(x,y,z) \mid x^{\alpha}y^{\alpha} \geq |z|, x \geq 0, y \geq 0 \}`.

Support for power cone constraints is a recent addition (v1.1.8), and CVXPY currently does
not have any atoms that take advantage of this constraint. If you want you want to use this
type of constraint in your model, you will need to instantiate ``PowCone3D`` and/or ``PowConeND``
objects manually.

MIP - :ref:`Mixed-Integer Programming <Mixed-integer programs>` refers to problems where some decision variables are constrained to be integer values.

By default CVXPY calls the solver most specialized to the problem type. For example, `ECOS`_ is called for SOCPs.
`SCS`_ can handle all problems (except mixed-integer programs). If the problem is a QP, CVXPY will use `OSQP`_.

You can change the solver called by CVXPY using the ``solver`` keyword argument. If the solver you choose cannot solve the problem, CVXPY will raise an exception. Here's example code solving the same problem with different solvers.

.. code-block:: python

    # Solving a problem with different solvers.
    x = cp.Variable(2)
    obj = cp.Minimize(x[0] + cp.norm(x, 1))
    constraints = [x >= 2]
    prob = cp.Problem(obj, constraints)

    # Solve with OSQP.
    prob.solve(solver=cp.OSQP)
    print("optimal value with OSQP:", prob.value)

    prob.solve(solver=cp.CLARABEL)
    print("optimal value with CLARABEL:", prob.value)

    # Solve with {solver_name}
    prob.solve(solver=cp.{solver_name})
    print("optimal value with {solver_name}:", prob.value)

    optimal value with OSQP: 6.0
    ...
    optimal value with CLARABEL: 6.0

Use the ``installed_solvers`` utility function to get a list of the solvers your installation of CVXPY supports.

.. code:: python

    print(installed_solvers())

::

    ['CBC', 'CVXOPT', 'MOSEK', 'GLPK', 'GLPK_MI', 'ECOS', 'SCS', 'SDPA'
     'SCIPY', 'GUROBI', 'OSQP', 'CPLEX', 'NAG', 'SCIP', 'XPRESS', 'PROXQP']

Viewing solver output
^^^^^^^^^^^^^^^^^^^^^

All the solvers can print out information about their progress while solving the problem. This information can be useful in debugging a solver error. To see the output from both CVXPY and the solvers, set ``verbose=True`` in the solve method. If you want to see the output from the solver only, set ``solver_verbose=True``.

.. code:: python

    # Solve with ECOS and display output.
    prob.solve(solver=cp.ECOS, verbose=True)
    print(f"optimal value with ECOS: {prob.value}")

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
For more information, see the :doc:`DGP tutorial </tutorial/dgp/index>`.

Solver stats
------------

When the ``solve`` method is called on a problem object and a solver is invoked,
the problem object records the optimal value, the values of the primal and dual variables,
and several solver statistics.
We have already discussed how to view the optimal value and variable values.
The solver statistics are accessed via the ``problem.solver_stats`` attribute,
which returns a :class:`~cvxpy.problems.problem.SolverStats` object.
For example, ``problem.solver_stats.solve_time`` gives the time it took the solver to solve the problem.

.. note::

    Information stored in ``problem.solver_stats`` differs in the solver used.
    For example, if we use ``MOSEK``, ``problem.solver_stats.num_iters`` includes the following: ``iinfitem.intpnt_iter``, ``liinfitem.simplex_iter``
    or ``iinfitem.mio_num_relax``. In addition, ``problem.solver_stats.extra_stats`` includes ``liinfitem.mio_intpnt_iter`` and ``liinfitem.mio_simplex_iter``.
    For more information, please visit https://docs.mosek.com/latest/pythonapi/constants.html

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
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)),
                       [x >= 0])

    b.value = numpy.random.randn(m)
    prob.solve()
    print("First solve time:", prob.solver_stats.solve_time)

    b.value = numpy.random.randn(m)
    prob.solve(warm_start=True)
    print("Second solve time:", prob.solver_stats.solve_time)

::

   First solve time: 11.14
   Second solve time: 2.95

The speed up in this case comes from caching the KKT matrix factorization.
If ``A`` were a parameter, factorization caching would not be possible and the benefit of
warm start would only be a good initial point.

Warm start can also be used to provide an initial guess the first time a problem is solved.
The initial guess is constructed from the ``value`` field of the problem variables.
If the same problem is solved a second time, the initial guess is constructed from the
cached previous solution as described above (rather than from the ``value`` field).

.. _solveropts:

Setting solver options
----------------------

The `OSQP`_, `ECOS`_, `GLOP`_, `MOSEK`_, `CBC`_, `CVXOPT`_, `NAG`_, `PDLP`_, `GUROBI`_, `SCS`_ , `CLARABEL`_, `DAQP`_, `PIQP`_ and `PROXQP`_ Python interfaces allow you to set solver options such as the maximum number of iterations. You can pass these options along through CVXPY as keyword arguments.

For example, here we tell SCS to use an indirect method for solving linear equations rather than a direct method.

.. code:: python

    # Solve with SCS, use sparse-indirect method.
    prob.solve(solver=cp.SCS, verbose=True, use_indirect=True)
    print(f"optimal value with SCS: {prob.value}")

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

.. info:: `OSQP`_ options:
   :collapsible: open

    ``'max_iter'``
        maximum number of iterations (default: 10,000).

    ``'eps_abs'``
        absolute accuracy (default: 1e-5).

    ``'eps_rel'``
        relative accuracy (default: 1e-5).

    For others see `OSQP documentation <https://osqp.org/docs/interfaces/solver_settings.html>`_.

.. info:: `PROXQP`_ options:
   :collapsible:

    ``'backend'``
        solver backend [dense, sparse] (default: dense).

    ``'max_iter'``
        maximum number of iterations (default: 10,000).

    ``'eps_abs'``
        absolute accuracy (default: 1e-8).

    ``'eps_rel'``
        relative accuracy (default: 0.0).

    ``'rho'``
        primal proximal parameter (default: 1e-6).

    ``'mu_eq'``
        dual equality constraint proximal parameter (default: 1e-3).

    ``'mu_in'``
        dual inequality constraint proximal parameter (default: 1e-1).

.. info:: `ECOS`_ options:
   :collapsible:

    ``'max_iters'``
        maximum number of iterations (default: 100).

    ``'abstol'``
        absolute accuracy (default: 1e-8).

    ``'reltol'``
        relative accuracy (default: 1e-8).

    ``'feastol'``
        tolerance for feasibility conditions (default: 1e-8).

    ``'abstol_inacc'``
        absolute accuracy for inaccurate solution (default: 5e-5).

    ``'reltol_inacc'``
        relative accuracy for inaccurate solution (default: 5e-5).

    ``'feastol_inacc'``
        tolerance for feasibility condition for inaccurate solution (default: 1e-4).

.. info:: `DAQP`_ options:
   :collapsible:

    For more information `see the DAQP documentation <https://darnstrom.github.io/daqp/parameters/>`_,
    some features of DAQP are currently unsupported in CVXPY.

    ``'primal_tol'``
        tolerance for primal infeasibility (default: 1e-6).
    ``'dual_tol'``
        olerance for dual infeasibility (default: 1e-12).
    ``'zero_tol'``
        values below are regarded as zero (default: 1e-11).
    ``'pivot_tol'``
        value used for determining if rows in the LDL factorization should be exchanged.
        A higher value improves stability (default: 1e-6).
    ``'progress_tol'``
        minimum change in objective function to consider it progress (default: 1e-6).
    ``'cycle_tol'``
        allowed number of iterations without progress before terminating (default: 10).
    ``'iter_limit'``
        maximum number of iterations before terminating (default: 1000).
    ``'fval_bound'``
        Maximum allowed objective function value. The solver terminates if the dual
        objective exceeds this value (since it is a lower bound of the optimal value,
        default: 1e30).
    ``'eps_prox'``
        Regularization parameter used for proximal-point iterations (0 means that
        no proximal-point iterations are performed). If the
        cost matrix has a null eigenvalue, setting this to 0 (upstream's default)
        makes DAQP fail. Note that CVXPY's canonicalization procedure may add extra
        variables with 0 quadratic cost which cause the cost matrix to have null eigenvalues
        (default: 1e-5 if there are null eigenvalues, else 0).
    ``'eta_prox'``
        Tolerance that determines if a fix-point has been reached during
        proximal-point iterations (default: 1e-6).

.. info:: `GLOP`_ options:
   :collapsible:

    ``'time_limit_sec'``
        Time limit for the solve, in seconds.

    ``'parameters_proto'``
        A `ortools.glop.parameters_pb2.GlopParameters` protocol buffer message.
        For the definition of GlopParameters, see
        `here <https://github.com/google/or-tools/blob/2cb85b4eead4c38e1c54b48044f92087cf165bce/ortools/glop/parameters.proto#L26>`_.

.. info:: `MOSEK`_ options
   :collapsible:

    ``'mosek_params'``
        A dictionary of MOSEK parameters in the form ``name: value``. Parameter names
        should be strings, as in the MOSEK C API or command line, for example
        ``'MSK_DPAR_BASIS_TOL_X'``, ``'MSK_IPAR_NUM_THREADS'`` etc. Values are strings,
        integers or floats, depending on the parameter.
        See `example <https://docs.mosek.com/latest/faq/faq.html#cvxpy>`_.

    ``'save_file'``
        The name of a file where MOSEK will save the problem just before optimization.
        Refer to MOSEK documentation for a list of supported file formats. File format
        is chosen based on the extension.

    ``'bfs'``
        For a linear problem, if ``bfs=True``, then the basic solution will be retrieved
        instead of the interior-point solution. This assumes no specific MOSEK
        parameters were used which prevent computing the basic solution.

    ``'accept_unknown'``
        If ``accept_unknown=True``, an inaccurate solution will be returned, even if
        it is arbitrarily bad, when the solver does not generate an optimal
        point under the given conditions.

    ``'eps'``
        Applies tolerance ``eps`` to termination parameters for (conic) interior-point,
        simplex, and MIO solvers. The full list of termination parameters is returned
        by ``MOSEK.tolerance_params()`` in
        ``cvxpy.reductions.solvers.conic_solvers.mosek_conif``.
        Explicitly defined parameters take precedence over ``eps``.


    .. note::

        In CVXPY 1.1.6 we did a complete rewrite of the MOSEK interface. The main
        takeaway is that we now dualize all continuous problems. The dualization is
        automatic because this eliminates the previous need for a large number of
        slack variables, and never results in larger problems compared to our old
        MOSEK interface. If you notice MOSEK solve times are slower for some of your
        problems under CVXPY 1.1.6 or higher, be sure to use the MOSEK solver options
        to tell MOSEK that it should solve the dual; this can be accomplished by
        adding the ``(key, value)`` pair ``('MSK_IPAR_INTPNT_SOLVE_FORM', 'MSK_SOLVE_DUAL')``
        to the ``mosek_params`` argument.

.. info:: `CVXOPT`_ options
   :collapsible:

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
        Controls the method used to solve systems of linear equations at each step of CVXOPT's
        interior-point algorithm. This parameter can be a string (with one of several values),
        or a function handle.

        KKT solvers built-in to CVXOPT can be specified by strings  'ldl', 'ldl2', 'qr', 'chol',
        and 'chol2'. If 'chol' is chosen, then CVXPY will perform an additional presolve
        procedure to eliminate redundant constraints. You can also set ``kktsolver='robust'``.
        The 'robust' solver is implemented in python, and is part of CVXPY source code; the
        'robust' solver doesn't require a presolve phase to eliminate redundant constraints,
        however it can be slower than 'chol'.

        Finally, there is an option to pass a function handle for the ``kktsolver`` argument.
        Passing a KKT solver based on a function handle allows you to take complete control of
        solving the linear systems encountered in CVXOPT's interior-point algorithm. The API for
        KKT solvers of this form is a small wrapper around CVXOPT's API for function-handle KKT
        solvers. The precise API that CVXPY users are held to is described in the CVXPY source
        code: `cvxpy/reductions/solvers/kktsolver.py <https://github.com/cvxpy/cvxpy/blob/master/cvxpy/reductions/solvers/kktsolver.py>`_.

.. info:: `SDPA`_ options
   :collapsible:

    ``'maxIteration'``
        The maximum number of iterations. (default: 100).

    ``'epsilonStar'``
        The accuracy of an approximate optimal solution for primal and dual SDP. (default: 1.0E-7).

    ``'lambdaStar'``
        An initial point. (default: 1.0E2).

    ``'omegaStar'``
        The search region for an optimal solution. (default: 2.0).

    ``'lowerBound'``
        Lower bound of the minimum objective value of the primal SDP. (default: -1.0E5).

    ``'upperBound'``
        Upper bound of the maximum objective value of the dual SDP. (default: 1.0E5).

    ``'betaStar'``
        The parameter for controlling the search direction if the current point is feasible. (default: 0.1).

    ``'betaBar'``
        The parameter for controlling the search direction if the current point is infeasible. (default: 0.2).

    ``'gammaStar'``
        A reduction factor for the primal and dual step lengths. (default: 0.9).

    ``'epsilonDash'``
        The relative accuracy of an approximate optimal solution between primal and dual SDP. (default: 1.0E-7).

    ``'isSymmetric'``
        Specify whether to check the symmetricity of input matrices. (default: False).

    ``'isDimacs'``
        Specify whether to compute DIMACS ERROR. (default: False).

    ``'numThreads'``
        numThreads (default: ``'multiprocessing.cpu_count()'``).

    ``'domainMethod'``
        Algorithm option for exploiting sparsity in the domain space. Can be ``'none'`` (exploiting no sparsity in the domain space) or ``'basis'`` (using basis representation) (default: ``'none'``).

    ``'rangeMethod'``
        Algorithm option for exploiting sparsity in the range space. Can be ``'none'`` (exploiting no sparsity in the range space) or ``'decomp'`` (using matrix decomposition) (default: ``'none'``).

    ``'frvMethod'``
        The method to eliminate free variables. Can be ``'split'`` or ``'elimination'`` (default: ``'split'``).

    ``'rho'``
        The parameter of range in split method or pivoting in elimination method. (default: 0.0).

    ``'zeroPoint'``
        The zero point of matrix operation, determine unboundness, or LU decomposition. (default: 1.0E-12).

.. info:: `SCS`_ options
   :collapsible:

    ``'max_iters'``
        maximum number of iterations (default: 2500).

    ``'eps'``
        convergence tolerance (default: 1e-4).

    ``'alpha'``
        relaxation parameter (default: 1.8).


    ``'acceleration_lookback'``
        Anderson Acceleration parameter for SCS 2.0 and higher. This can be any positive or negative integer;
        its default value is 10. See `this page of the SCS documentation <https://www.cvxgrp.org/scs/algorithm/acceleration.html#in-scs>`_
        for more information.

        .. warning::
            The value of this parameter often effects whether or not SCS 2.X will converge to an accurate solution.
            If you don't *explicitly* set ``acceleration_lookback`` and SCS 2.X fails to converge, then CVXPY
            will raise a warning and try to re-solve the problem with ``acceleration_lookback=0``.
            No attempt will be made to re-solve with problem if you have SCS version 3.0 or higher.

    ``'scale'``
        balance between minimizing primal and dual residual (default: 5.0).

    ``'normalize'``
        whether to precondition data matrices (default: True).

    ``'use_indirect'``
        whether to use indirect solver for KKT sytem (instead of direct) (default: True).

    ``'use_quad_obj'``
        whether to use a quadratic objective or reduce it to SOC constraints (default: True).

.. info:: `CBC`_ options
   :collapsible:

    Cut-generation through `CGL`_

    General remarks:
        - some of these cut-generators seem to be buggy (observed problems with AllDifferentCuts, RedSplitCuts, LandPCuts, PreProcessCuts)
        - a few of these cut-generators will generate noisy output even if ``'verbose=False'``

    The following cut-generators are available:
        ``GomoryCuts``, ``MIRCuts``, ``MIRCuts2``, ``TwoMIRCuts``, ``ResidualCapacityCuts``, ``KnapsackCuts`` ``FlowCoverCuts``, ``CliqueCuts``, ``LiftProjectCuts``, ``AllDifferentCuts``, ``OddHoleCuts``, ``RedSplitCuts``, ``LandPCuts``, ``PreProcessCuts``, ``ProbingCuts``, ``SimpleRoundingCuts``.

    ``'CutGenName'``
        if cut-generator is activated (e.g. ``'GomoryCuts=True'``)

    ``'integerTolerance'``
        an integer variable is deemed to be at an integral value if it is no further than this value (tolerance) away

    ``'maximumSeconds'``
        stop after given amount of seconds

    ``'maximumNodes'``
        stop after given maximum number of nodes

    ``'maximumSolutions'``
        stop after evalutation x number of solutions

    ``'numberThreads'``
        sets the number of threads

    ``'allowableGap'``
        returns a solution if the gap between the best known solution and the best possible solution is less than this value.

    ``'allowableFractionGap'``
        returns a solution if the gap between the best known solution and the best possible solution is less than this fraction.

    ``'allowablePercentageGap'``
        returns if the gap between the best known solution and the best possible solution is less than this percentage.

.. info:: `COPT`_ options:
   :collapsible:

    COPT solver options are specified in CVXPY as keyword arguments. The full list of COPT parameters with defaults is listed `here <https://guide.coap.online/copt/en-doc/index.html#parameters>`_.

.. info:: `CPLEX`_ options:
   :collapsible:

    ``'cplex_params'``
        a dictionary where the key-value pairs are composed of parameter names (as used in the CPLEX Python API) and parameter values. For example, to set the advance start switch parameter (i.e., CPX_PARAM_ADVIND), use "advance" for the parameter name. For the data consistency checking and modeling assistance parameter (i.e., CPX_PARAM_DATACHECK), use "read.datacheck" for the parameter name, and so on.

    ``'cplex_filename'``
        a string specifying the filename to which the problem will be written. For example, use "model.lp", "model.sav", or "model.mps" to export to the LP, SAV, and MPS formats, respectively.

    ``reoptimize``
        A boolean. This is only relevant for problems where CPLEX initially produces an "infeasible or unbounded" status.
        Its default value is False. If set to True, then if CPLEX produces an "infeasible or unbounded" status, its algorithm
        parameters are automatically changed and the problem is re-solved in order to determine its precise status.

.. info:: `NAG`_ options:
   :collapsible:

    ``'nag_params'``
        a dictionary of NAG option parameters. Refer to NAG's Python or Fortran API for details. For example, to set the maximum number of iterations for a linear programming problem to 20, use "LPIPM Iteration Limit" for the key name and 20 for its value .

.. info:: SCIP_ options:
   :collapsible:

    ``'scip_params'`` a dictionary of SCIP optional parameters, a full list of parameters with defaults is listed `here <https://www.scipopt.org/doc-5.0.1/html/PARAMETERS.php>`_.

.. info:: `SCIPY`_ options:
   :collapsible:

    ``'scipy_options'`` a dictionary of SciPy optional parameters, a full list of parameters with defaults is listed `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html#scipy.optimize.linprog>`_.

    * **Please note**: All options should be listed as key-value pairs within the ``'scipy_options'`` dictionary, and there should not be a nested dictionary called options. Some of the methods have different parameters, so please check the parameters for the method you wish to use, e.g., for method = 'highs-ipm'. Also, note that the 'integrality' and 'bounds' options should never be specified within ``'scipy_options'`` and should instead be specified using CVXPY.

    * The main advantage of this solver is its ability to use the `HiGHS`_ LP and MIP solvers, which are coded in C++. However, these require versions of SciPy larger than 1.6.1 and 1.9.0, respectively. To use the `HiGHS`_ LP solvers, simply set the method parameter to 'highs-ds' (for dual-simplex), 'highs-ipm' (for interior-point method) or 'highs' (which will choose either 'highs-ds' or 'highs-ipm' for you). To use the `HiGHS`_ MIP solver, leave the method parameter unspecified or set it explicitly to 'highs'.

.. info:: `PDLP`_ options:
   :collapsible:

    ``'time_limit_sec'``
        Time limit for the solve, in seconds.

    ``'parameters_proto'``
        A `ortools.pdlp.solvers_pb2.PrimalDualHybridGradientParams` protocol buffer message.
        For the definition of PrimalDualHybridGradientParams, see
        `here <https://github.com/google/or-tools/blob/a3ef28e824ee84a948796dffbb8254e67714cb56/ortools/pdlp/solvers.proto#L150>`_.

.. info:: `GUROBI`_ options:
   :collapsible:

    Gurobi solver options are specified in CVXPY as keyword arguments. The full list of Gurobi parameters with defaults is listed `here <https://www.gurobi.com/documentation/9.1/refman/parameters.html>`_.

    In addition to Gurobi's parameters, the following options are available:

    ``'env'``
        Allows for the passage of a Gurobi Environment, which specifies parameters and license information.  Keyword arguments will override any settings in this environment.

    ``reoptimize``
        A boolean. This is only relevant for problems where GUROBI initially produces an "infeasible or unbounded" status.
        Its default value is False. If set to True, then if GUROBI produces an "infeasible or unbounded" status, its algorithm
        parameters are automatically changed and the problem is re-solved in order to determine its precise status.

.. info:: `CLARABEL`_ options:
   :collapsible:

    ``'max_iter'``
        maximum number of iterations (default: 50).

    ``'time_limit'``
        time limit in seconds (default: 0.0, giving no limit).

    For others see `CLARABEL documentation <https://oxfordcontrol.github.io/ClarabelDocs/stable/api_settings/>`_.

.. info::  `XPRESS`_ options:
   :collapsible:

    ``'save_iis'``
        Whether (and how many) Irreduceable Infeasible Subsystems
        (IISs) should be saved in the event a problem is found to be
        infeasible. If 0 (default), no IIS is saved; if negative, all
        IISs are stored; if a positive ``'k>0'``, at most ``'k'`` IISs
        are saved.

    ``'write_mps'``
        Filename (with extension ``'.mps'``) in which Xpress will save
        the quadratic or conic problem.

    ``'maxtime'``
        Time limit in seconds (must be integer).

    All controls of the Xpress Optimizer can be specified within the ``'solve'``
    command. For all controls see `FICO Xpress Optimizer manual <https://www.fico.com/fico-xpress-optimization/docs/dms2019-03/solver/optimizer/HTML/chapter7.html>`_.

.. info:: `PIQP`_ options:
   :collapsible:

    ``'backend'``
        solver backend [dense, sparse] (default: sparse).

    ``'max_iter'``
        maximum number of iterations (default: 250).

    ``'eps_abs'``
        absolute accuracy (default: 1e-8).

    ``'eps_rel'``
        relative accuracy (default: 1e-9).

    For others see `PIQP documentation <https://predict-epfl.github.io/piqp/interfaces/settings>`_.

Custom Solvers
------------------------------------
Although ``cvxpy`` supports many different solvers out of the box, it is also possible to define and use custom solvers. This can be helpful in prototyping or developing custom solvers tailored to a specific application.

To do so, you have to implement a solver class that is a child of ``cvxpy.reductions.solvers.qp_solvers.qp_solver.QpSolver`` or ``cvxpy.reductions.solvers.conic_solvers.conic_solver.ConicSolver``. Then you pass an instance of this solver class to ``solver.solve(.)`` as following:

.. code:: python3

    import cvxpy as cp
    from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP


    class CUSTOM_OSQP(OSQP):
        MIP_CAPABLE=False

        def name(self):
            return "CUSTOM_OSQP"

        def solve_via_data(self, *args, **kwargs):
            print("Solving with a custom QP solver!")
            super().solve_via_data(*args, **kwargs)


    x = cp.Variable()
    quadratic = cp.square(x)
    problem = cp.Problem(cp.Minimize(quadratic))
    problem.solve(solver=CUSTOM_OSQP())

You might also want to override the methods ``invert`` and ``import_solver`` of the ``Solver`` class.

Note that the string returned by the ``name`` property should be different to all of the officially supported solvers
(a list of which can be found in ``cvxpy.settings.SOLVERS``). Also, if your solver is mixed integer capable,
you should set the class variable ``MIP_CAPABLE`` to ``True``. If your solver is both mixed integer capable
and a conic solver (as opposed to a QP solver), you should set the class variable ``MI_SUPPORTED_CONSTRAINTS``
to the list of cones supported when solving mixed integer problems. Usually ``MI_SUPPORTED_CONSTRAINTS``
will be the same as the class variable ``SUPPORTED_CONSTRAINTS``.

.. _CVXOPT: http://cvxopt.org/
.. _COPT: https://github.com/COPT-Public/COPT-Release
.. _ECOS: https://www.embotech.com/ECOS
.. _SCS: http://github.com/cvxgrp/scs
.. _SDPA: https://sdpa-python.github.io
.. _DAQP: https://darnstrom.github.io/daqp/
.. _GLOP: https://developers.google.com/optimization
.. _GLPK: https://www.gnu.org/software/glpk/
.. _GLPK_MI: https://www.gnu.org/software/glpk/
.. _GUROBI: https://www.gurobi.com/
.. _MOSEK: https://www.mosek.com/
.. _CBC: https://projects.coin-or.org/Cbc
.. _CGL: https://projects.coin-or.org/Cgl
.. _CPLEX: https://www.ibm.com/docs/en/icos
.. _NAG: https://www.nag.co.uk/nag-library-python/
.. _OSQP: https://osqp.org/
.. _PDLP: https://developers.google.com/optimization
.. _SCIP: https://scip.zib.de/
.. _XPRESS: https://www.fico.com/en/products/fico-xpress-optimization
.. _SCIPY: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html#scipy.optimize.linprog
.. _HiGHS: https://www.maths.ed.ac.uk/hall/HiGHS/#guide
.. _CLARABEL: https://oxfordcontrol.github.io/ClarabelDocs/
.. _PIQP: https://predict-epfl.github.io/piqp/
.. _PROXQP: https://github.com/simple-robotics/proxsuite
