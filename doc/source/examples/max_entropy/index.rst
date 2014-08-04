.. _max_entropy:

Entropy Maximization
====================

A derivative work by Judson Wilson, 6/2/2014. Adapted from the CVX
example of the same name, by Joëlle Skaf, 4/24/2008.

Introduction
------------

Consider the linear inequality constrained entropy maximization problem:

.. math::

   \begin{array}{ll}
       \mbox{maximize}   & -\sum_{i=1}^n x_i \log(x_i) \\
       \mbox{subject to} & \sum_{i=1}^n x_i = 1 \\
                         & Fx \succeq g,
       \end{array}

where the variable is :math:`x \in \mathbf{{\mbox{R}}}^{n}`.

This problem can be formulated in CVXPY using the ``entr`` atom.

Generate problem data
---------------------

.. code:: python

    import cvxpy as cvx
    import numpy as np

    # Make random input repeatable.
    np.random.seed(0)

    # Matrix size parameters.
    n = 20
    m = 10
    p = 5

    # Generate random problem data.
    tmp = np.mat(np.random.rand(n, 1))
    A = np.mat(np.random.randn(m, n))
    b = A*tmp
    F = np.mat(np.random.randn(p, n))
    g = F*tmp + np.mat(np.random.rand(p, 1))

Formulate and solve problem
---------------------------

.. code:: python

    # Entropy maximization.
    x = cvx.Variable(n)
    obj = cvx.Maximize(cvx.sum_entries(cvx.entr(x)))
    constraints = [A*x == b,
                   F*x <= g ]
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.CVXOPT, verbose=True)

    # Print result.
    print "\nThe optimal value is:", prob.value
    print '\nThe optimal solution is:'
    print x.value

.. parsed-literal::

         pcost       dcost       gap    pres   dres
     0:  0.0000e+00 -7.5220e+00  2e+01  1e+00  1e+00
     1: -6.0720e+00 -5.9875e+00  2e+00  1e-01  2e-01
     2: -5.4688e+00 -5.5885e+00  4e-01  2e-02  5e-02
     3: -5.4595e+00 -5.4889e+00  5e-02  2e-03  2e-02
     4: -5.4763e+00 -5.4816e+00  1e-02  3e-04  7e-03
     5: -5.4804e+00 -5.4809e+00  1e-03  4e-05  2e-03
     6: -5.4809e+00 -5.4809e+00  3e-05  1e-06  4e-04
     7: -5.4809e+00 -5.4809e+00  4e-07  1e-08  2e-05
     8: -5.4809e+00 -5.4809e+00  4e-09  1e-10  3e-07
     9: -5.4809e+00 -5.4809e+00  4e-11  1e-12  5e-09
    Optimal solution found.

    The optimal value is: 5.48090148635

    The optimal solution is:
    [[ 0.43483319]
     [ 0.66111715]
     [ 0.49201039]
     [ 0.36030618]
     [ 0.38416629]
     [ 0.30283658]
     [ 0.41730232]
     [ 0.79107794]
     [ 0.76667302]
     [ 0.38292365]
     [ 1.2479328 ]
     [ 0.50416987]
     [ 0.68053832]
     [ 0.67163958]
     [ 0.13877259]
     [ 0.5248668 ]
     [ 0.08418897]
     [ 0.56927148]
     [ 0.50000248]
     [ 0.78291311]]

