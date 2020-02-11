
Mixed-integer quadratic program
===============================

A mixed-integer quadratic program (MIQP) is an optimization problem of
the form

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & x^T Q x + q^T x + r \\
       \mbox{subject to} & x \in \mathcal{C}\\
       & x \in \mathbf{Z}^n,
       \end{array}

where :math:`x \in \mathbf{Z}^n` is the optimization variable
(:math:`\mathbf Z^n` is the set of :math:`n`-dimensional vectors with
integer-valued components), :math:`Q \in \mathbf{S}_+^n` (the set of
:math:`n \times n` symmetric positive semidefinite matrices),
:math:`q \in \mathbf{R}^n`, and :math:`r \in \mathbf{R}` are problem
data, and :math:`\mathcal C` is some convex set.

An example of an MIQP is mixed-integer least squares, which has the form

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & \|Ax-b\|_2^2 \\
       \mbox{subject to} & x \in \mathbf{Z}^n,
       \end{array}

where :math:`x \in \mathbf{Z}^n` is the optimization variable, and
:math:`A \in \mathbf{R}^{m \times n}` and :math:`b \in \mathbf{R}^{m}`
are the problem data. A solution :math:`x^{\star}` of this problem will
be a vector in :math:`\mathbf Z^n` that minimizes :math:`\|Ax-b\|_2^2`.

Example
-------

In the following code, we solve a mixed-integer least-squares problem
with CVXPY.

.. code:: python

    import cvxpy as cp
    import numpy as np

.. code:: python

    # Generate a random problem
    np.random.seed(0)
    m, n= 40, 25
    
    A = np.random.rand(m, n)
    b = np.random.randn(m)

.. code:: python

    # Construct a CVXPY problem
    x = cp.Variable(n, integer=True)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    prob = cp.Problem(objective)
    prob.solve()




.. parsed-literal::

    13.66000322824753



.. code:: python

    print("Status: ", prob.status)
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)


.. parsed-literal::

    Status:  optimal
    The optimal value is 13.66000322824753
    A solution x is
    [-1.  1.  1. -1.  0.  0. -1. -2.  0.  0.  0.  1.  1.  0.  1.  0. -1. -1.
     -1.  0.  2. -1.  2.  0. -1.]

