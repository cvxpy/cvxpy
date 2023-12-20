.. _advanced:

Advanced Features
=================

This section of the tutorial covers features of CVXPY intended for users with advanced knowledge of convex optimization. We recommend `Convex Optimization <https://www.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe as a reference for any terms you are unfamiliar with.

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

<<<<<<< HEAD
<<<<<<< HEAD
.. _attributes:

Attributes
----------

Variables and parameters can be created with attributes specifying additional properties.
For example, ``Variable(nonneg=True)`` is a scalar variable constrained to be nonnegative.
Similarly, ``Parameter(nonpos=True)`` is a scalar parameter constrained to be nonpositive.
The full constructor for :py:class:`Leaf <cvxpy.expressions.leaf.Leaf>` (the parent class
of :py:class:`Variable <cvxpy.expressions.variable.Variable>` and
:py:class:`Parameter <cvxpy.expressions.constants.parameter.Parameter>`) is given below.

.. function:: Leaf(shape=None, value=None, nonneg=False, nonpos=False, complex=False, imag=False, symmetric=False, diag=False, PSD=False, NSD=False, hermitian=False, boolean=False, integer=False, sparsity=None, pos=False, neg=False)

    Creates a Leaf object (e.g., Variable or Parameter).
    Only one attribute can be active (set to True).

    :param shape: The variable dimensions (0D by default). Cannot be more than 2D.
    :type shape: tuple or int
    :param value: A value to assign to the variable.
    :type value: numeric type
    :param nonneg: Is the variable constrained to be nonnegative?
    :type nonneg: bool
    :param nonpos: Is the variable constrained to be nonpositive?
    :type nonpos: bool
    :param complex: Is the variable constrained to be complex-valued?
    :type complex: bool
    :param imag: Is the variable constrained to be imaginary?
    :type imag: bool
    :param symmetric: Is the variable constrained to be symmetric?
    :type symmetric: bool
    :param diag: Is the variable constrained to be diagonal?
    :type diag: bool
    :param PSD: Is the variable constrained to be symmetric positive semidefinite?
    :type PSD: bool
    :param NSD: Is the variable constrained to be symmetric negative semidefinite?
    :type NSD: bool
    :param hermitian: Is the variable constrained to be Hermitian?
    :type hermitian: bool
    :param boolean:
        Is the variable boolean (i.e., 0 or 1)? True, which constrains
        the entire variable to be boolean, False, or a list of
        indices which should be constrained as boolean, where each
        index is a tuple of length exactly equal to the
        length of shape.
    :type boolean: bool or list of tuple
    :param integer: Is the variable integer? The semantics are the same as the boolean argument.
    :type integer: bool or list of tuple
    :param sparsity: Fixed sparsity pattern for the variable.
    :type sparsity: list of tuplewith
    :param pos: Is the variable constrained to be positive?
    :type pos: bool
    :param neg: Is the variable constrained to be negative?
    :type neg: bool
    :param bounds: Lower and upper bounds.
    :type bounds: An iterable of length two.

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

One downside of using attributes over explicit constraints is that dual variables will not be recorded. Dual variable values
are only recorded for explicit constraints.

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

CVXPY provides interfaces to many mixed-integer solvers, including open source and commercial solvers.
For licensing reasons, CVXPY does not install any of the preferred solvers by default.

The preferred open source mixed-integer solvers in CVXPY are GLPK_MI_, CBC_ and SCIP_. The CVXOPT_
python package provides CVXPY with access to GLPK_MI; CVXOPT can be installed by running
``pip install cvxopt`` in your command line or terminal. SCIP supports nonlinear models, but
GLPK_MI and CBC do not.

CVXPY comes with ECOS_BB -- an open source mixed-integer nonlinear solver -- by default. However
ECOS_BB will not be called automatically; you must explicitly call ``prob.solve(solver='ECOS_BB')``
if you want to use it (:ref:`changed in CVXPY 1.1.6 <changes116>`). This policy stems from the fact
that there are recurring correctness issues with ECOS_BB. If you rely on this solver for some
application then you need to be aware of the increased risks that come with using it.
If you need to use an open-source mixed-integer nonlinear solver from CVXPY, then we recommend you install SCIP.

If you need to solve a large mixed-integer problem quickly, or if you have a nonlinear mixed-integer
model that is challenging for SCIP, then you will need to use a commercial solver such as CPLEX_,
GUROBI_, XPRESS_, MOSEK_, or COPT_. Commercial solvers require licenses to run. CPLEX, GUROBI, and MOSEK
provide free licenses to those
in academia (both students and faculty), as well as trial versions to those outside academia.
CPLEX Free Edition is available at no cost regardless of academic status, however it still requires
online registration, and it's limited to problems with at most 1000 variables and 1000 constraints.
XPRESS has a free community edition which does not require registration, however it is limited
to problems where the sum of variables count and constraint count does not exceed 5000.
COPT also has a free community edition that is limited to problems with at most 2000 variables 
and 2000 constraints.

.. note::
   If you develop an open-source mixed-integer solver with a permissive license such
   as Apache 2.0, and you're interested in incorporating your solver into CVXPY's default installation,
   please reach out to us at our `GitHub issues <https://github.com/cvxpy/cvxpy/issues>`_. We are
   particularly interested in incorporating a simple mixed-integer SOCP solver.

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

The top-level expressions in the problem objective must be real valued,
but subexpressions may be complex.
Arithmetic and all linear atoms are defined for complex expressions.
The nonlinear atoms ``abs`` and all norms except ``norm(X, p)`` for ``p < 1`` are also defined for complex expressions.
All atoms whose domain is symmetric matrices are defined for Hermitian matrices.
Similarly, the atoms ``quad_form(x, P)`` and ``matrix_frac(x, P)`` are defined for complex ``x`` and Hermitian ``P``.
All constraints are defined for complex expressions.

The following additional atoms are provided for working with complex expressions:

* ``real(expr)`` gives the real part of ``expr``.
* ``imag(expr)`` gives the imaginary part of ``expr`` (i.e., ``expr = real(expr) + 1j*imag(expr)``).
* ``conj(expr)`` gives the complex conjugate of ``expr``.
* ``expr.H`` gives the Hermitian (conjugate) transpose of ``expr``.

=======
>>>>>>> c4aab172d2cb5c4a49d6a826dbd5bd50f5e02f0b
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
   expr = cp.transforms.indicator(constraints)
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

Getting the standard form
-------------------------

If you are interested in getting the standard form that CVXPY produces for a
problem, you can use the ``get_problem_data`` method. When a problem is solved, 
a :class:`~cvxpy.reductions.solvers.solving_chain.SolvingChain` passes a
low-level representation that is compatible with the targeted solver to a
solver, which solves the problem. This method returns that low-level
representation, along with a ``SolvingChain`` and metadata for unpacking
a solution into the problem. This low-level representation closely resembles,
but is not identical to, the arguments supplied to the solver.

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
  probdata, _, _ = problem.get_problem_data(cp.SCS)

  import scs
  data = {
    'A': probdata['A'],
    'b': probdata['b'],
    'c': probdata['c'],
  }
  cone_dims = probdata['dims']
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

.. _canonicalization-backends:

Canonicalization backends
------------------------------------
Users can select from multiple canonicalization backends by adding the ``canon_backend``
keyword argument to the ``.solve()`` call, e.g. ``problem.solve(canon_backend=cp.SCIPY_CANON_BACKEND)``
(Introduced in CVXPY 1.3).
This can speed up the canonicalization time significantly for some problems.
Currently, the following canonicalization backends are supported:

*  CPP (default): The original C++ implementation, also referred to as CVXCORE.
*  | SCIPY: A pure Python implementation based on the SciPy sparse module.
   | Generally fast for problems that are already vectorized.
*  NUMPY: Reference implementation in pure NumPy. Fast for some small or dense problems.
