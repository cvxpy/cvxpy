
Least-squares
=============

In a least-squares, or linear regression, problem, we have measurements
:math:`A \in \mathcal{R}^{m \times n}` and :math:`b \in \mathcal{R}^m`
and seek a vector :math:`x \in \mathcal{R}^{n}` such that :math:`Ax` is
close to :math:`b`. Closeness is defined as the sum of the squared
differences:

.. math::  \sum_{i=1}^m (a_i^Tx - b_i)^2, 

also known as the :math:`\ell_2`-norm squared, :math:`\|Ax - b\|_2^2`.

For example, we might have a dataset of :math:`m` users, each
represented by :math:`n` features. Each row :math:`a_i^T` of :math:`A`
is the features for user :math:`i`, while the corresponding entry
:math:`b_i` of :math:`b` is the measurement we want to predict from
:math:`a_i^T`, such as ad spending. The prediction is given by
:math:`a_i^Tx`.

We find the optimal :math:`x` by solving the optimization problem

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & \|Ax - b\|_2^2.
       \end{array}

Let :math:`x^\star` denote the optimal :math:`x`. The quantity
:math:`r = Ax^\star - b` is known as the residual. If
:math:`\|r\|_2 = 0`, we have a perfect fit.

Example
-------

In the following code, we solve a least-squares problem with CVXPY.

.. code:: python

    # Import packages.
    import cvxpy as cp
    import numpy as np
    
    # Generate data.
    m = 20
    n = 15
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    
    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    cost = cp.sum_squares(A @ x - b)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    
    # Print result.
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)
    print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)


.. parsed-literal::

    
    The optimal value is 7.005909828287484
    The optimal x is
    [ 0.17492418 -0.38102551  0.34732251  0.0173098  -0.0845784  -0.08134019
      0.293119    0.27019762  0.17493179 -0.23953449  0.64097935 -0.41633637
      0.12799688  0.1063942  -0.32158411]
    The norm of the residual is  2.6468679280023557

