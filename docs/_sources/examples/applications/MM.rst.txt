
Method of multipliers
=====================

The method of multipliers is an algorithm for solving convex
optimization problems. Suppose we have a problem of the form

.. raw:: latex

   \begin{array}{ll}
   \mbox{minimize} & f(x)\\
   \mbox{subject to} & Ax = b,
   \end{array}

where :math:`f` is convex, :math:`x \in \mathbf{R}^n` is the
optimization variable, and :math:`A \in \mathbf{R}^{m \times n}` and
:math:`b \in \mathbf{R}^m` are problem data.

To apply the method of multipliers, we first form the augmented
Lagrangian

.. math:: L_{\rho}(x,y) = f(x) + y^T(Ax - b) + (\rho/2)\|Ax-b\|^2_2.

The dual function associated with the augmented Lagrangian is
:math:`g_{\rho}(y) = \inf_x L_{\rho}(x,y)`. The dual function
:math:`g_{\rho}(y)` is concave and its maximal value is the same as the
optimal value of the original problem.

We maximize the dual function using gradient ascent. Each step of
gradient ascent reduces to the :math:`x` and :math:`y` updates

.. raw:: latex

   \begin{array}{lll}
   x^{k+1} & := & \mathop{\rm argmin}_{x}\left(f(x) + (y^k)^T(Ax - b) + (\rho/2)\left\|Ax-b\right\|^2_2 \right) \\
   y^{k+1} & := & y^{k} + \rho(Ax^{k+1}-b)
   \end{array}

The following CVXPY script implements the method of multipliers and uses
it to solve an optimization problem.

.. code:: 

    from cvxpy import *
    import numpy as np
    np.random.seed(1)
    
    # Initialize data.
    MAX_ITERS = 10
    rho = 1.0
    n = 20
    m = 10
    A = np.random.randn(m,n)
    b = np.random.randn(m,1)
    
    # Initialize problem.
    x = Variable(shape=(n,1))
    f = norm(x, 1)
    
    # Solve with CVXPY.
    Problem(Minimize(f), [A*x == b]).solve()
    print "Optimal value from CVXPY", f.value
    
    # Solve with method of multipliers.
    resid = A*x - b
    y = Parameter(shape=(m,1)); y.value = np.zeros(m)
    aug_lagr = f + y.T*resid + (rho/2)*sum_squares(resid)
    for t in range(MAX_ITERS):
        Problem(Minimize(aug_lagr)).solve()
        y.value += rho*resid.value
        
    print "Optimal value from method of multipliers", f.value


.. parsed-literal::

    Optimal value from CVXPY 5.57311224497
    Optimal value from method of multipliers 5.57276155042

