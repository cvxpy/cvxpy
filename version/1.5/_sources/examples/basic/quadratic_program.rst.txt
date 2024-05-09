
Quadratic program
=================

A quadratic program is an optimization problem with a quadratic
objective and affine equality and inequality constraints. A common
standard form is the following:

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & (1/2)x^TPx + q^Tx\\
       \mbox{subject to} & Gx \leq h \\
                         & Ax = b.
       \end{array}

Here :math:`P \in \mathcal{S}^{n}_+`, :math:`q \in \mathcal{R}^n`,
:math:`G \in \mathcal{R}^{m \times n}`, :math:`h \in \mathcal{R}^m`,
:math:`A \in \mathcal{R}^{p \times n}`, and :math:`b \in \mathcal{R}^p`
are problem data and :math:`x \in \mathcal{R}^{n}` is the optimization
variable. The inequality constraint :math:`Gx \leq h` is elementwise.

A simple example of a quadratic program arises in finance. Suppose we
have :math:`n` different stocks, an estimate :math:`r \in \mathcal{R}^n`
of the expected return on each stock, and an estimate
:math:`\Sigma \in \mathcal{S}^{n}_+` of the covariance of the returns.
Then we solve the optimization problem

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & (1/2)x^T\Sigma x - r^Tx\\
       \mbox{subject to} & x \geq 0 \\
                         & \mathbf{1}^Tx = 1,
       \end{array}

to find a portfolio allocation :math:`x \in \mathcal{R}^n_+` that
optimally balances expected return and variance of return.

When we solve a quadratic program, in addition to a solution
:math:`x^\star`, we obtain a dual solution :math:`\lambda^\star`
corresponding to the inequality constraints. A positive entry
:math:`\lambda^\star_i` indicates that the constraint
:math:`g_i^Tx \leq h_i` holds with equality for :math:`x^\star` and
suggests that changing :math:`h_i` would change the optimal value.

Example
-------

In the following code, we solve a quadratic program with CVXPY.

.. code:: python

    # Import packages.
    import cvxpy as cp
    import numpy as np
    
    # Generate a random non-trivial quadratic program.
    m = 15
    n = 10
    p = 5
    np.random.seed(1)
    P = np.random.randn(n, n)
    P = P.T @ P
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)
    
    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [G @ x <= h,
                      A @ x == b])
    prob.solve()
    
    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)


.. parsed-literal::

    
    The optimal value is 86.89141585569918
    A solution x is
    [-1.68244521  0.29769913 -2.38772183 -2.79986015  1.18270433 -0.20911897
     -4.50993526  3.76683701 -0.45770675 -3.78589638]
    A dual solution corresponding to the inequality constraints is
    [ 0.          0.          0.          0.          0.         10.45538054
      0.          0.          0.         39.67365045  0.          0.
      0.         20.79927156  6.54115873]

