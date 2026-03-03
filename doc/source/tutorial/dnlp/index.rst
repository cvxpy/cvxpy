.. _dnlp:

Disciplined Nonlinear Programming
=================================

Disciplined nonlinear programming (DNLP) is a system for constructing nonlinear
programs (NLPs) with rules similar to those of disciplined convex programming (DCP).
DNLP extends DCP by allowing smooth functions to be freely mixed with nonsmooth convex
and concave functions, with rules governing how nonsmooth functions can be used.

CVXPY lets you form and solve DNLP problems. For example, the following code
solves a simple (nonconvex) nonlinear program:

.. code:: python

    import cvxpy as cp
    import numpy as np

    # problem data
    np.random.seed(0)
    n = 3
    A = np.random.randn(n, n)
    A = A.T @ A

    # formulate optimization problem
    x = cp.Variable(n)
    obj = cp.Maximize(cp.quad_form(x, A))
    constraints = [cp.sum_squares(x) == 1]

    # initialize and solve
    x.value = np.ones(n)
    prob = cp.Problem(obj, constraints)
    prob.solve(nlp=True)
    print("Optimal value from DNLP:", prob.value)

    # the optimal value can also be found via the maximum eigenvalue of A
    eigenvalues = np.linalg.eigvalsh(A)
    print("Maximum eigenvalue:     ", np.max(eigenvalues))

Note that for CVXPY to treat the problem as an NLP, you must pass the option ``nlp=True`` to the
``solve()`` method.

.. warning::
    In convex optimization and DCP, solvers are guaranteed to find globally optimal solutions.
    In contrast, when solving a nonconvex NLP, there are no guarantees of finding globally optimal solutions,
    or even locally optimal solutions. An NLP solver may converge to an infeasible point, even if the problem 
    is feasible. Furthermore, the solution returned by an NLP solver may depend on the initial point provided.
    Specifying a good initial point (by setting the ``value`` attribute of
    the variables) can significantly improve convergence.

For an in-depth reference on DNLP, see our
`accompanying paper <https://web.stanford.edu/~boyd/papers/dnlp.html>`_.

Atoms and expressions
---------------------

DNLP classifies atoms into three categories: **smooth**, **nonsmooth convex**, and
**nonsmooth concave**. A `full list of new DNLP atoms <dnlp-atoms_>`_ is presented
at the end of this page.

DNLP classifies expressions based on the types of atoms they contain and how those atoms 
are composed. There are three expression types: **smooth**, **linearizable convex (L-convex)**, and 
**linearizable concave (L-concave)**. If you are familiar with DCP, there are very 
compact definitions of these expression types in terms of the DCP curvature types:

-  A smooth expression is an expression that only consists of smooth atoms.
   Smooth is the analog of affine in DCP.
-  An expression is linearizable convex (L-convex) if it is DCP convex when 
   all smooth atoms in the expression are treated as affine.
-  An expression is linearizable concave (L-concave) if it is DCP concave when all 
   smooth atoms in the expression are treated as affine.

You can check expression classifications using the methods ``expr.is_smooth()``, 
``expr.is_linearizable_convex()``, and ``expr.is_linearizable_concave()``. 
Note that smooth expressions are both L-convex and L-concave.

It is also possible to define the DNLP expression types without reference to DCP.
For more details, see Section 3.2 of the `DNLP paper <https://web.stanford.edu/~boyd/papers/dnlp.html>`_.

DNLP problems
--------------

A DNLP problem minimizes an L-convex objective or maximizes an L-concave objective.
The valid constraint types are:

-  smooth ``==`` smooth
-  L-convex ``<=`` L-concave
-  L-concave ``>=`` L-convex

You can check that a problem satisfies the DNLP rules by calling
``problem.is_dnlp()``. CVXPY will raise an exception if you call
``problem.solve(nlp=True)`` on a non-DNLP problem.

.. _dnlp-atoms:

DNLP atoms
----------

In DNLP, all atoms from the :ref:`DCP atom library <functions>` are available and
classified as smooth if they are twice continuously differentiable on the interior of
their domain. Convex and concave DCP atoms that are not smooth (such as :ref:`abs <abs>`,
:ref:`maximum <maximum>`, :ref:`norm1 <norm1>`, :ref:`minimum <minimum>`, :ref:`min <min>`, etc.)
retain their convexity/concavity and can appear in L-convex and L-concave expressions respectively.

Some existing CVXPY atoms gain new meaning in DNLP. These are summarized in the table below.
For example, in DCP, the ``multiply`` atom requires that one of the arguments is a constant,
but in DNLP, ``multiply`` is smooth and can be used with two variable arguments. 

.. list-table::
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Monotonicity

   * - multiply(x, y)

     - :math:`xy`
     - :math:`x, y \in \mathbf{R}`
     - depends on sign

   * - quad_form(x, Q)

       :math:`Q \in \mathbf{S}^n`
     - :math:`x^T Q x`
     - :math:`x \in \mathbf{R}^n`
     - depends on sign

In addition, DNLP introduces the following new smooth atoms that are neither convex
nor concave. These atoms can only be used in DNLP problems.

.. list-table::
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Monotonicity

   * - sin(x)

     - :math:`\sin(x)`
     - :math:`x \in \mathbf{R}`
     - none

   * - cos(x)

     - :math:`\cos(x)`
     - :math:`x \in \mathbf{R}`
     - none

   * - tan(x)

     - :math:`\tan(x)`
     - :math:`x \in (-\pi/2, \pi/2)`
     - none

   * - sinh(x)

     - :math:`(e^x - e^{-x})/2`
     - :math:`x \in \mathbf{R}`
     - incr.

   * - tanh(x)

     - :math:`(e^x - e^{-x})/(e^x + e^{-x})`
     - :math:`x \in \mathbf{R}`
     - incr.

   * - asinh(x)

     - :math:`\ln(x + \sqrt{x^2 + 1})`
     - :math:`x \in \mathbf{R}`
     - incr.

   * - atanh(x)

     - :math:`\frac{1}{2} \ln \frac{1+x}{1-x}`
     - :math:`x \in (-1, 1)`
     - incr.

   * - sigmoid(x)

     - :math:`\frac{1}{1 + e^{-x}}`
     - :math:`x \in \mathbf{R}`
     - incr.
