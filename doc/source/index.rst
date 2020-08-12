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

This short script is a basic example of what CVXPY can do; in addition
to convex programming, CVXPY also supports a generalization of geometric
programming.

For a guided tour of CVXPY, check out the :doc:`tutorial
</tutorial/index>`. Browse the :doc:`library of examples
</examples/index>` for applications to machine learning, control, finance, and
more. 

**News.**

* CVXPY v1.1.a4 has been released. This pre-release
  adds support for differentiating through DCP and DGP
  problems. See the tutorial on :ref:`derivatives`,
  and the `accompanying <https://web.stanford.edu/~boyd/papers/diff_cvxpy.html>`_
  `papers <https://web.stanford.edu/~boyd/papers/diff_dgp.html>`_,
  for more information. Install it using ``pip install --pre --upgrade cvxpy``.

* CVXPY v1.1.a1 (which is an alpha version) has been released. This version makes
  repeatedly canonicalizing parametrized problems much faster than before. See
  the tutorial on :ref:`dpp` for more information.

* CVXPY v1.0.24 supports
  `disciplined quasiconvex programming <https://web.stanford.edu/~boyd/papers/dqcp.html>`_,
  which lets you formulate and solve quasiconvex programs.
  See the :doc:`tutorial </tutorial/dqcp/index>` for more information.

* CVXPY v1.0.11 supports
  `disciplined geometric programming <https://web.stanford.edu/~boyd/papers/dgp.html>`_,
  which lets you formulate geometric programs and log-log convex programs.
  See the :doc:`tutorial </tutorial/dgp/index>` for more information.

* CVXPY 1.0 brings the API closer to NumPy
  and the architecture closer to software compilers, making it easy for
  developers to write custom problem transformations and target custom solvers.
  CVXPY 1.0 is not backwards compatible with previous versions
  of CVXPY. For more details, see :ref:`updates`.

**Solvers.**

CVXPY relies on the open source solvers `ECOS`_, `OSQP`_, and `SCS`_.
Additional solvers are supported, but must be installed separately. For
background on convex optimization, see the book `Convex Optimization
<http://www.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe.

**Development.**

CVXPY began as a Stanford University :doc:`research project </citing/index>`.
Today, CVXPY is a community project, built from the contributions of many
researchers and engineers.

CVXPY is developed and maintained by
`Steven Diamond <http://web.stanford.edu/~stevend2/>`_ and
`Akshay Agrawal <https://akshayagrawal.com>`_, with many others contributing
significantly. A non-exhaustive list of people who have shaped CVXPY over the
years includes Stephen Boyd, Eric Chu, Robin Verschueren, Bartolomeo Stellato,
Riley Murray, Jaehyun Park, Enzo Busseti, AJ Friend, Judson Wilson, and Chris
Dembia.

We appreciate all contributions. To get involved, see our :doc:`contributing
guide </contributing/index>`.

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
