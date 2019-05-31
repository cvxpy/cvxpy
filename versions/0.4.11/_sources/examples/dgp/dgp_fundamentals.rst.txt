
DGP fundamentals
================

This notebook will introduce you to the fundamentals of **disciplined
geometric programming** (**DGP**), which lets you formulate and solve
*log-log convex programs* (LLCPs) in CVXPY.

LLCPs are problems that become convex after the variables, objective
functions, and constraint functions are replaced with their logs, an
operation that we refer to as a log-log transformation. LLCPs generalize
geometric programming.

.. code:: python

    import cvxpy as cp

1. Log-log curvature
====================

Just as every Expression in CVXPY has a curvature (constant, affine,
convex, concave, or unknown), every Expression also has a log-log
curvature.

A function :math:`f : D \subseteq \mathbf{R}^{n}_{++} \to \mathbf{R}` is
said to be **log-log convex** if the function :math:`F(u)=\log f(e^u)`,
with domain :math:`\{u \in \mathbf{R}^n : e^u \in D\}`, is convex (where
:math:`\mathbf{R}^{n}_{++}` denotes the set of positive reals and the
logarithm and exponential are meant elementwise); the function :math:`F`
is called the log-log transformation of :math:`f`. The function
:math:`f` is **log-log concave** if :math:`F` is concave, and it is
**log-log affine** if :math:`F` is affine.

Notice that if a function has log-log curvature, then its domain and
range can only include positive numbers.

The log-log curvature of an ``Expression`` can be accessed via its
``.log_log_curvature`` attribute. For an ``Expression`` to have known
log-log curvature, all of the ``Constant``\ s, ``Variable``\ s, and
``Parameter``\ s it refers to must be elementwise positive.

.. code:: python

    # Only elementwise positive constants are allowed in DGP.
    c = cp.Constant(1.0)
    print(c, c.log_log_curvature)
    
    c = cp.Constant([1.0, 2.0])
    print(c, c.log_log_curvature)
    
    c = cp.Constant([1.0, 0.0])
    print(c, c.log_log_curvature)
    
    c = cp.Constant(-2.0)
    print(c, c.log_log_curvature)


.. parsed-literal::

    1.0 LOG-LOG CONSTANT
    [1. 2.] LOG-LOG CONSTANT
    [1. 0.] UNKNOWN
    -2.0 UNKNOWN


.. code:: python

    # Variables and parameters must be positive, ie, they must be constructed with the option `pos=True`
    v = cp.Variable(pos=True)
    print(v, v.log_log_curvature)
    
    v = cp.Variable()
    print(v, v.log_log_curvature)
    
    p = cp.Parameter(pos=True)
    print(p, p.log_log_curvature)
    
    p = cp.Parameter()
    print(p, p.log_log_curvature)


.. parsed-literal::

    var0 LOG-LOG AFFINE
    var1 UNKNOWN
    param2 LOG-LOG CONSTANT
    param3 UNKNOWN


Functions from geometric programming
------------------------------------

A function :math:`f(x)` is log-log affine if and only if it is given by

.. math::


   f(x) = cx_1^{a_1}x_2^{a_2} \ldots x_n^{a_n},

where :math:`c > 0` and the :math:`a_i` are real numbers. In the context
of geometric programming, such a function is called a monomial.

.. code:: python

    x = cp.Variable(shape=(3,), pos=True, name="x")
    c = 2.0
    a = [0.5, 2.0, 1.8]
    
    monomial = c * x[0] ** a[0] * x[1] ** a[1] * x[2] ** a[2]
    # Monomials are not convex.
    assert not monomial.is_convex()
    
    # They are, however, log-log affine.
    print(monomial, ":", monomial.log_log_curvature)
    assert monomial.is_log_log_affine()


.. parsed-literal::

    2.0 * power(x[0], 1/2) * power(x[1], 2) * power(x[2], 9/5) : LOG-LOG AFFINE


A sum of monomial functions is log-log convex; in the context of
geometric programming, such a function is called a posynomial. There are
functions that are not posynomials that are still log-log convex.

.. code:: python

    x = cp.Variable(pos=True, name="x")
    y = cp.Variable(pos=True, name="y")
    
    constant = cp.Constant(2.0)
    monomial = constant * x * y
    posynomial = monomial + (x ** 1.5) * (y ** -1)
    reciprocal = posynomial ** -1
    unknown = reciprocal + posynomial
    
    print(constant, ":", constant.log_log_curvature)
    print(monomial, ":", monomial.log_log_curvature)
    print(posynomial, ":", posynomial.log_log_curvature)
    print(reciprocal, ":", reciprocal.log_log_curvature)
    print(unknown, ":", unknown.log_log_curvature)


.. parsed-literal::

    2.0 : LOG-LOG CONSTANT
    2.0 * x * y : LOG-LOG AFFINE
    2.0 * x * y + power(x, 3/2) * power(y, -1) : LOG-LOG CONVEX
    power(2.0 * x * y + power(x, 3/2) * power(y, -1), -1) : LOG-LOG CONCAVE
    power(2.0 * x * y + power(x, 3/2) * power(y, -1), -1) + 2.0 * x * y + power(x, 3/2) * power(y, -1) : UNKNOWN


2. Log-log curvature ruleset
============================

CVXPY has a library of atomic functions with known log-log curvature and
monotonicty. It uses this information to tag every ``Expression``, i.e.,
every composition of atomic functions, with a log-log curvature. In
particular,

A function :math:`f(expr_1,expr_2,...,expr_n)` is log-log convex if
:math:`f` is a log-log convex function and for each expri one of the
following conditions holds:

:math:`f` is increasing in argument i and :math:`expr_i` is log-log
convex. :math:`f` is decreasing in argument :math:`i` and :math:`expr_i`
is log-log concave. :math:`expr_i` is log-log affine. A function
:math:`f(expr_1,expr_2,...,expr_n)` is log-log concave if :math:`f` is a
log-log concave function and for each :math:`expr_i` one of the
following conditions holds:

:math:`f` is increasing in argument :math:`i` and :math:`expr_i` is
log-log concave. :math:`f` is decreasing in argument :math:`i` and
:math:`expr_i` is log-log convex. :math:`expr_i` is log-log affine. A
function :math:`f(expr_1,expr_2,...,expr_n)` is log-log affine if
:math:`f` is an log-log affine function and each :math:`expr_i` is
log-log affine.

If none of the three rules apply, the expression
:math:`f(expr_1,expr_2,...,expr_n)` is marked as having unknown
curvature.

If an Expression satisfies the composition rule, we colloquially say
that the ``Expression`` “is DGP.” You can check whether an
``Expression`` is DGP by calling the method ``is_dgp()``.

.. code:: python

    x = cp.Variable(pos=True, name="x")
    y = cp.Variable(pos=True, name="y")
    
    monomial = 2.0 * x * y
    posynomial = monomial + (x ** 1.5) * (y ** -1)
    
    print(monomial, "is dgp?", monomial.is_dgp())
    print(posynomial, "is dgp?", posynomial.is_dgp())


.. parsed-literal::

    2.0 * x * y is dgp? True
    2.0 * x * y + power(x, 3/2) * power(y, -1) is dgp? True


3. DGP problems
===============

An LLCP is an optimization problem of the form

.. math::


   \begin{equation}
   \begin{array}{ll}
   \mbox{minimize} & f_0(x) \\
   \mbox{subject to} & f_i(x) \leq \tilde{f_i}, \quad i=1, \ldots, m\\
   & g_i(x) = \tilde{g_i}, \quad i=1, \ldots, p,
   \end{array}
   \end{equation}

where the functions :math:`f_i` are log-log convex, :math:`\tilde{f_i}`
are log-log concave, and the functions :math:`g_i` and
:math:`\tilde{g_i}` are log-log affine. An optimization problem with
constraints of the above form in which the goal is to maximize a log-log
concave function is also an LLCP.

A problem is DGP if additionally all the functions are DGP. You can
check whether a CVXPY ``Problem`` is DGP by calling its ``.is_dgp()``
method.

.. code:: python

    x = cp.Variable(pos=True, name="x")
    y = cp.Variable(pos=True, name="y")
    z = cp.Variable(pos=True, name="z")
    
    objective_fn = x * y * z
    constraints = [
      4 * x * y * z + 2 * x * z <= 10, x <= 2*y, y <= 2*x, z >= 1]
    assert objective_fn.is_log_log_concave()
    assert all(constraint.is_dgp() for constraint in constraints)
    problem = cp.Problem(cp.Maximize(objective_fn), constraints)
    
    print(problem)
    print("Is this problem DGP?", problem.is_dgp())


.. parsed-literal::

    maximize x * y * z
    subject to 4.0 * x * y * z + 2.0 * x * z <= 10.0
               x <= 2.0 * y
               y <= 2.0 * x
               1.0 <= z
    Is this problem DGP? True


Solving DGP problems
--------------------

You can solve a DGP ``Problem`` by calling its ``solve`` method with
``gp=True``.

.. code:: python

    problem.solve(gp=True)
    print("Optimal value:", problem.value)
    print(x, ":", x.value)
    print(y, ":", y.value)
    print(z, ":", z.value)
    print("Dual values: ", list(c.dual_value for c in constraints))


.. parsed-literal::

    Optimal value: 1.9999999926890524
    x : 0.9999999989968756
    y : 1.9999999529045318
    z : 1.000000020895385
    Dual values:  [1.1111111199586956, 1.94877846244994e-09, 0.1111111217156332, 0.11111112214962586]


If you forget to supply ``gp=True``, an error will be raised.

.. code:: python

    try:
        problem.solve()
    except cp.DCPError as e:
        print(e)


.. parsed-literal::

    Problem does not follow DCP rules. However, the problem does follow DGP rules. Consider calling this function with `gp=True`.


4. Next steps
=============

Atoms
-----

CVXPY has a large library of log-log convex functions, including common
functions like :math:`\exp`, :math:`\log`, and the difference between
two numbers. Check out the tutorial on our website for the full list of
atoms: https://www.cvxpy.org/tutorial/dgp/index.html

References
----------

For a reference on DGP, consult the following paper:
https://web.stanford.edu/~boyd/papers/dgp.html
