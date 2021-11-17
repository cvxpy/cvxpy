
Minimum-length least squares
============================

This notebook shows how to solve a *minimum-length least squares*
problem, which finds a minimum-length vector :math:`x \in \mathbf{R}^n`
achieving small mean-square error (MSE) for a particular least squares
problem:

.. math::

   \begin{array}{ll}
   \mbox{minimize} & \mathrm{len}(x) \\
   \mbox{subject to} & \frac{1}{n}\|Ax - b\|_2^2 \leq \epsilon,
   \end{array}

where the variable is :math:`x` and the problem data are :math:`n`,
:math:`A`, :math:`b`, and :math:`\epsilon`.

This is a quasiconvex program (QCP). It can be specified using
disciplined quasiconvex programming
(`DQCP <https://www.cvxpy.org/tutorial/dqcp/index.html>`__), and it can
therefore be solved using CVXPY.

.. code:: 

    !pip install --upgrade cvxpy

.. code:: 

    import cvxpy as cp
    import numpy as np

The below cell constructs the problem data.

.. code:: 

    n = 10
    np.random.seed(1)
    A = np.random.randn(n, n)
    x_star = np.random.randn(n)
    b = A @ x_star
    epsilon = 1e-2

And the next cell constructs and solves the QCP.

.. code:: 

    x = cp.Variable(n)
    mse = cp.sum_squares(A @ x - b)/n
    problem = cp.Problem(cp.Minimize(cp.length(x)), [mse <= epsilon])
    print("Is problem DQCP?: ", problem.is_dqcp())
    
    problem.solve(qcp=True)
    print("Found a solution, with length: ", problem.value)


.. parsed-literal::

    Is problem DQCP?:  True
    Found a solution, with length:  8.0


.. code:: 

    print("MSE: ", mse.value)


.. parsed-literal::

    MSE:  0.00926009328813662


.. code:: 

    print("x: ", x.value)


.. parsed-literal::

    x:  [-2.58366030e-01  1.38434327e+00  2.10714108e-01  9.44811159e-01
     -1.14622208e+00  1.51283929e-01  6.62931941e-01 -1.16358584e+00
      2.78132907e-13 -1.76314786e-13]


.. code:: 

    print("x_star: ", x_star)


.. parsed-literal::

    x_star:  [-0.44712856  1.2245077   0.40349164  0.59357852 -1.09491185  0.16938243
      0.74055645 -0.9537006  -0.26621851  0.03261455]

