.. cvxpy documentation master file, created by
   sphinx-quickstart on Mon Jan 27 20:47:07 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CVXPY
========================

CVXPY is a Python-embedded modeling language for convex optimization problems.

For example, the following code solves a least-squares problem where the variable is constrained by lower and upper bounds:

.. code:: python

    from cvxpy import *
    import numpy

    # Problem data.
    m = 30
    n = 20
    numpy.random.seed(1)
    A = numpy.random.randn(m, n)
    b = numpy.random.randn(m)

    # Construct the problem.
    x =Variable(n)
    objective =M inimize(sum_squares(A*x - b)))
    constraints = [0 <= x, x <= 1]
    prob = Problem(objective, constraints)

    # The optimal objective is returned by p.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    print x.value
    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    print constraints[0].dual_value

Tutorial
--------

.. toctree::
    :maxdepth: 2

    intro/index
    dcp/index
    functions/index
