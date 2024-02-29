.. cvxpy documentation master file, created by
   sphinx-quickstart on Mon Jan 27 20:47:07 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CVXPY 1.5
====================

.. meta::
   :description: An open source Python-embedded modeling language for convex optimization problems.
                 Express your problem in a natural way that follows the math.
   :keywords: convex optimization, open source, software,

.. raw:: html

      <script type="application/ld+json">
      {
         "@context" : "https://schema.org",
         "@type" : "WebSite",
         "name" : "CVXPY",
         "url" : "https://www.cvxpy.org/"
      }
      </script>

**Convex optimization, for everyone.**

*We are building a CVXPY community* `on Discord <https://discord.gg/4urRQeGBCr>`_. *Join the conversation!*

CVXPY is an open source Python-embedded modeling language for convex
optimization problems. It lets you express your problem in a natural way that
follows the math, rather than in the restrictive standard form required by
solvers.

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
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    print(x.value)
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    print(constraints[0].dual_value)

This short script is a basic example of what CVXPY can do. In addition
to convex programming, CVXPY also supports a generalization of geometric
programming, mixed-integer convex programs, and quasiconvex programs.

For a guided tour of CVXPY, check out the :doc:`tutorial
</tutorial/index>`. For applications to machine learning, control, finance, and
more, browse the :doc:`library of examples </examples/index>`. For
background on convex optimization, see the book `Convex Optimization
<https://www.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe.

CVXPY relies on the open source solvers `Clarabel`_, `OSQP`_, `SCS`_, and `ECOS`_.
Additional solvers are supported, but must be installed separately.

**Community.**

The CVXPY community consists of researchers, data scientists, software
engineers, and students from all over the world. We welcome you to join us!

* To chat with the CVXPY community in real-time, join us `on Discord <https://discord.gg/4urRQeGBCr>`_.

* To have longer, in-depth discussions with the CVXPY community, use `Github discussions <https://github.com/cvxpy/cvxpy/discussions>`_.

* To share feature requests and bug reports, use the `issue tracker <https://github.com/cvxpy/cvxpy/issues>`_.

**Development.**

CVXPY is a community project, built from the contributions of many
researchers and engineers.

CVXPY is developed and maintained by
`Steven Diamond <https://stevendiamond.me/>`_,
`Akshay Agrawal <https://akshayagrawal.com>`_,
`Riley Murray <https://rileyjmurray.wordpress.com/>`_,
`Philipp Schiele <https://www.philippschiele.com/>`_, and
`Bartolomeo Stellato <https://stellato.io/>`_ with many others contributing
significantly. A non-exhaustive list of people who have shaped CVXPY over the
years includes Stephen Boyd, Eric Chu, Robin Verschueren,
Jaehyun Park, Enzo Busseti, AJ Friend, Judson Wilson, Chris Dembia, and
Philipp Schiele.

We appreciate all contributions. To get involved, see our :doc:`contributing
guide </contributing/index>` and join us `on Discord <https://discord.gg/4urRQeGBCr>`_.

**News.**

CVXPY 1.3 introduced the option for users to specify different canonicalization backends,
which can drastically reduce the canonicalization time. Initially, a second backend based on
the SciPy sparse module was added. Read more about the new backends here: 
:ref:`canonicalization-backends`. See `CVXPYgen <https://github.com/cvxgrp/cvxpygen>`_ for a
complementary code generation approach. Following the introduction of semantic versioning,
since the CVXPY 1.3 release, everything that can be imported from the `cvxpy` namespace is
considered to be part of the public API.

.. _Clarabel: https://github.com/oxfordcontrol/Clarabel.rs
.. _OSQP: https://osqp.org/
.. _ECOS: http://github.com/ifa-ethz/ecos
.. _SCS: http://github.com/cvxgrp/scs

.. toctree::
   :hidden:

   install/index

.. toctree::
    :hidden:

    User Guide <tutorial/index>

.. toctree::
   :hidden:

   API Documentation <api_reference/cvxpy>

.. toctree::
   :hidden:

   examples/index

.. toctree::
   :hidden:

   contributing/index

.. toctree::
   :hidden:

   Changelog <updates/index>

.. toctree::
   :maxdepth: 1
   :hidden:

   faq/index

.. toctree::
   :hidden:

   resources/index

