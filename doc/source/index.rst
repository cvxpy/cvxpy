.. cvxpy documentation master file, created by
   sphinx-quickstart on Mon Jan 27 20:47:07 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CVXPY's documentation!
=================================

We can solve the following quadratic program:

.. math::

    \begin{array}{ll}
        \mbox{minimize} & \frac{1}{2} x^T P x + q^T x + r \\\\
        \mbox{subject to} & -1 \leq x_i \leq 1, i = 1, ..., n \\\\
    \end{array}

with just a few lines of code::

    import cvxopt
    import cvxpy as cp
    # Generate the data.
    n = 3
    P = cvxopt.matrix([	13, 12, -2,
    			12, 17,  6,
    			-2,  6, 12], (n, n))
    q = cvxopt.matrix([-22, -14.5, 13], (n, 1))
    r = 1

    # Frame and solve the problem.
    x = cp.Variable(n)
    objective = cp.Minimize(  0.5 * cp.quad_form(x, P)  + q.T * x + r )
    constraints = [ x >= -1, x <= 1]

    p = cp.Problem(objective, constraints)
    # The optimal objective is returned by p.solve().
    result = p.solve()

Contents:

.. toctree::
    :maxdepth: 1

    reference/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

