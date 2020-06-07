
Second-order cone program
=========================

A second-order cone program (SOCP) is an optimization problem of the
form

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & f^Tx\\
       \mbox{subject to} & \|A_ix + b_i\|_2 \leq c_i^Tx + d_i, \quad i=1,\ldots,m \\
                         & Fx = g,
       \end{array}

where :math:`x \in \mathcal{R}^{n}` is the optimization variable and
:math:`f \in \mathcal{R}^n`, :math:`A_i \in \mathcal{R}^{n_i \times n}`,
:math:`b_i \in \mathcal{R}^{n_i}`, :math:`c_i \in \mathcal{R}^n`,
:math:`d_i \in \mathcal{R}`, :math:`F \in \mathcal{R}^{p \times n}`, and
:math:`g \in \mathcal{R}^p` are problem data.

An example of an SOCP is the robust linear program

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & c^Tx\\
       \mbox{subject to} & (a_i + u_i)^Tx \leq b_i \textrm{ for all } \|u_i\|_2 \leq 1, \quad i=1,\ldots,m,
       \end{array}

where the problem data :math:`a_i` are known within an
:math:`\ell_2`-norm ball of radius one. The robust linear program can be
rewritten as the SOCP

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & c^Tx\\
       \mbox{subject to} & a_i^Tx + \|x\|_2 \leq b_i, \quad i=1,\ldots,m,
       \end{array}

When we solve a SOCP, in addition to a solution :math:`x^\star`, we
obtain a dual solution :math:`\lambda_i^\star` corresponding to each
second-order cone constraint. A non-zero :math:`\lambda_i^\star`
indicates that the constraint :math:`\|A_ix + b_i\|_2 \leq c_i^Tx + d_i`
holds with equality for :math:`x^\star` and suggests that
changing :math:`d_i` would change the optimal value.

Example
-------

In the following code, we solve a SOCP with CVXPY.

.. code:: python

    # Import packages.
    import cvxpy as cp
    import numpy as np
    
    # Generate a random feasible SOCP.
    m = 3
    n = 10
    p = 5
    n_i = 5
    np.random.seed(2)
    f = np.random.randn(n)
    A = []
    b = []
    c = []
    d = []
    x0 = np.random.randn(n)
    for i in range(m):
        A.append(np.random.randn(n_i, n))
        b.append(np.random.randn(n_i))
        c.append(np.random.randn(n))
        d.append(np.linalg.norm(A[i] @ x0 + b, 2) - c[i].T @ x0)
    F = np.random.randn(p, n)
    g = F @ x0
    
    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
          cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(f.T@x),
                      soc_constraints + [F @ x == g])
    prob.solve()
    
    # Print result.
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    for i in range(m):
        print("SOC constraint %i dual variable solution" % i)
        print(soc_constraints[i].dual_value)


.. parsed-literal::

    The optimal value is -9.582695716265503
    A solution x is
    [ 1.40303325  2.4194569   1.69146656 -0.26922215  1.30825472 -0.70834842
      0.19313706  1.64153496  0.47698583  0.66581033]
    SOC constraint 0 dual variable solution
    [ 0.61662526  0.35370661 -0.02327185  0.04253095  0.06243588  0.49886837]
    SOC constraint 1 dual variable solution
    [ 0.35283078 -0.14301082  0.16539699 -0.22027817  0.15440264  0.06571645]
    SOC constraint 2 dual variable solution
    [ 0.86510445 -0.114638   -0.449291    0.37810251 -0.6144058  -0.11377797]

