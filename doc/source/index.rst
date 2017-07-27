.. cvxpy documentation master file, created by
   sphinx-quickstart on Mon Jan 27 20:47:07 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CVXPY
================

**Join the** `CVXPY mailing list <https://groups.google.com/forum/#!forum/cvxpy>`_ **and** `Gitter chat <https://gitter.im/cvxgrp/cvxpy>`_ **for the best CVXPY support!**

**CVXPY 1.0 is under development**. **There will be some** `changes to the user interface <https://github.com/cvxgrp/cvxpy/issues/199>`_.

CVXPY is a Python-embedded modeling language for convex optimization problems. It allows you to express your problem in a natural way that follows the math, rather than in the restrictive standard form required by solvers.

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
    x = Variable(n)
    objective = Minimize(sum_squares(A*x - b))
    constraints = [0 <= x, x <= 1]
    prob = Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    print x.value
    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    print constraints[0].dual_value

This short script is a basic example of what CVXPY can do. CVXPY also supports simple ways to solve problems in parallel, higher-level abstractions such as object oriented convex optimization, and extensions for non-convex optimization.

CVXPY was designed and implemented by Steven Diamond, with input from Stephen Boyd and Eric Chu.

CVXPY was inspired by the MATLAB package `CVX <http://cvxr.com/cvx/>`_. See the book `Convex Optimization <http://www.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe for general background on convex optimization.

CVXPY relies on the open source solvers `ECOS`_, `CVXOPT`_, and `SCS`_.
Additional solvers are supported, but must be installed separately.

.. _CVXOPT: http://cvxopt.org/
.. _ECOS: http://github.com/ifa-ethz/ecos
.. _SCS: http://github.com/cvxgrp/scs

.. toctree::
   :hidden:

   install/index

.. toctree::
    :maxdepth: 3
    :hidden:

    tutorial/index

.. toctree::
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 1
   :hidden:

   faq/index

.. toctree::
   :hidden:

   citing/index

.. toctree::
   :hidden:

   community/index

.. toctree::
   :hidden:

   related_projects/index

.. toctree::
   :hidden:

   short_course/index

.. toctree::
   :hidden:

   license/index
