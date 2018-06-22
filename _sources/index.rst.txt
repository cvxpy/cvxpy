.. cvxpy documentation master file, created by
   sphinx-quickstart on Mon Jan 27 20:47:07 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CVXPY 1.0
====================

**Convex optimization, for everyone.**

*For the best support, join the* `CVXPY mailing list
<https://groups.google.com/forum/#!forum/cvxpy>`_ *and post your questions on*
`Stack Overflow <https://stackoverflow.com/questions/tagged/cvxpy>`_.

CVXPY is a Python-embedded modeling language for convex optimization problems.
It allows you to express your problem in a natural way that follows the math,
rather than in the restrictive standard form required by solvers.

For example, the following code solves a least-squares problem with box constraints:

.. code:: python

    import cvxpy as cp
    import numpy as np

    # Problem data.
    m = 30
    n = 20
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # Construct the problem.
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A*x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    print(x.value)
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    print(constraints[0].dual_value)

This short script is a basic example of what CVXPY can do. See the
:doc:`library of examples </examples/index>` for applications to machine
learning, control, finance, and more. For a guided tour of CVXPY, check out the
:doc:`tutorial </tutorial/index>`, and consult the :doc:`API reference
</api_reference/cvxpy>` for documentation about public symbols.

----------

CVXPY was designed and implemented by Steven Diamond, with input and
contributions from Stephen Boyd, Eric Chu, Akshay Agrawal, Robin Verschueren,
and many others; it was inspired by the MATLAB package `CVX
<http://cvxr.com/cvx/>`_.  Version 1.0 of CVXPY brings the API closer to NumPy
and the architecture closer to software compilers, making it easy for
developers to write custom problem transformations and target custom solvers.
Be aware that CVXPY 1.0 is not backwards compatible with previous versions
of CVXPY. For more details, see :ref:`updates`, which includes instructions
for migrating from previous versions of CVXPY.

CVXPY relies on the open source solvers `ECOS`_, `OSQP`_, and `SCS`_.
Additional solvers are supported, but must be installed separately. For
background on convex optimization, see the book `Convex Optimization
<http://www.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe.


.. _OSQP: https://osqp.org/
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
   :hidden:

   API Documentation <api_reference/cvxpy>

.. toctree::
   :maxdepth: 1
   :hidden:

   faq/index

.. toctree::
   :hidden:

   citing/index

.. toctree::
   :hidden:

   contributing/index

.. toctree::
   :hidden:

   related_projects/index

.. toctree::
   :hidden:

   updates/index

.. toctree::
   :hidden:

   short_course/index

.. toctree::
   :hidden:

   license/index

.. toctree::
   :hidden:

   versions/index
