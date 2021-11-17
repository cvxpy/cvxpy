
Linear program
==============

A linear program is an optimization problem with a linear objective and
affine inequality constraints. A common standard form is the following:

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & c^Tx \\
       \mbox{subject to} & Ax \leq b.
       \end{array}

Here :math:`A \in \mathcal{R}^{m \times n}`,
:math:`b \in \mathcal{R}^m`, and :math:`c \in \mathcal{R}^n` are problem
data and :math:`x \in \mathcal{R}^{n}` is the optimization variable. The
inequality constraint :math:`Ax \leq b` is elementwise.

For example, we might have :math:`n` different products, each
constructed out of :math:`m` components. Each entry :math:`A_{ij}` is
the amount of component :math:`i` required to build one unit of product
:math:`j`. Each entry :math:`b_i` is the total amount of component
:math:`i` available. We lose :math:`c_j` for each unit of product
:math:`j` (:math:`c_j < 0` indicates profit). Our goal then is to choose
how many units of each product :math:`j` to make, :math:`x_j`, in order
to minimize loss without exceeding our budget for any component.

In addition to a solution :math:`x^\star`, we obtain a dual solution
:math:`\lambda^\star`. A positive entry :math:`\lambda^\star_i`
indicates that the constraint :math:`a_i^Tx \leq b_i` holds with
equality for :math:`x^\star` and suggests that changing :math:`b_i`
would change the optimal value.

Example
-------

In the following code, we solve a linear program with CVXPY.

.. code:: python

    # Import packages.
    import cvxpy as cp
    import numpy as np
    
    # Generate a random non-trivial linear program.
    m = 15
    n = 10
    np.random.seed(1)
    s0 = np.random.randn(m)
    lamb0 = np.maximum(-s0, 0)
    s0 = np.maximum(s0, 0)
    x0 = np.random.randn(n)
    A = np.random.randn(m, n)
    b = A @ x0 + s0
    c = -A.T @ lamb0
    
    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c.T@x),
                     [A @ x <= b])
    prob.solve()
    
    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution is")
    print(prob.constraints[0].dual_value)


.. parsed-literal::

    
    The optimal value is -15.220912604467838
    A solution x is
    [-1.10131657 -0.16370661 -0.89711643  0.03228613  0.60662428 -1.12655967
      1.12985839  0.88200333  0.49089264  0.89851057]
    A dual solution is
    [0.         0.61175641 0.52817175 1.07296862 0.         2.3015387
     0.         0.7612069  0.         0.24937038 0.         2.06014071
     0.3224172  0.38405435 0.        ]

