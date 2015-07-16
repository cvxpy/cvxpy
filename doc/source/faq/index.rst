=====================================
FAQ
=====================================

.. contents::
  :local:
  :backlinks: none
  :depth: 1

Where can I get help with CVXPY?
--------------------------------
You can post questions about how to use CVXPY on the `CVXPY mailing list <https://groups.google.com/forum/#!forum/cvxpy>`_.
If you've found a bug in CVXPY or have a feature request,
create an issue on the `CVXPY Github issue tracker <https://github.com/cvxgrp/cvxpy/issues>`_.

Where can I learn more about convex optimization?
--------------------------------------------------
The book `Convex Optimization <http://web.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe is available for free online and has extensive background on convex optimization.
To learn more about disciplined convex programming,
visit the `DCP tutorial website <http://dcp.stanford.edu/>`_.

What do I do if I get a ``DCPError`` exception?
-----------------------------------------------
The problems you solve in CVXPY must follow the rules of disciplined convex programming (DCP).
DCP is like a type system for optimization problems.
For more about DCP, see the :ref:`DCP tutorial section <dcp>` or the `DCP tutorial website <http://dcp.stanford.edu/>`_.

How do I find DCP errors?
-------------------------
You can test whether a problem, objective, constraint, or expression satisfies the DCP
rules by calling ``object.is_dcp()``.
If the function returns ``False``,
there is a DCP error in that object.

What do I do if I get a ``SolverError`` exception?
--------------------------------------------------
Sometimes solvers encounter numerical issues and fail to solve a problem, in which case CVXPY raises a ``SolverError``.
If this happens to you,
try using different solvers on your problem,
as discussed in the "Choosing a solver" section of :ref:`Advanced Features <advanced>`.
If the solver CVXOPT fails, try using the solver option ``kktsolver=ROBUST_KKTSOLVER``.

What solvers does CVXPY support?
--------------------------------
See the "Solve method options" section in :ref:`Advanced Features <advanced>` for a list of the solvers CVXPY supports.
If you would like to use a solver CVXPY does not support,
make a feature request on the `CVXPY Github issue tracker <https://github.com/cvxgrp/cvxpy/issues>`_.

What are the differences between CVXPY's solvers?
-------------------------------------------------
The solvers support different classes of problems and occupy different points on the Pareto frontier of speed, accuracy, and open source vs. closed source.
See the "Solve method options" section in :ref:`Advanced Features <advanced>` for details.

What do I do if I get "RuntimeError: maximum recursion depth exceeded"?
------------------------------------------------------------------------
See `this thread <https://groups.google.com/forum/#!topic/cvxpy/btQuh4FsQ-I>`_ on the mailing list.

Can I use NumPy functions on CVXPY objects?
-------------------------------------------
No, you can only use CVXPY functions on CVXPY objects.
If you use a NumPy function on a CVXPY object,
it will probably fail in a confusing way.

Can I use SciPy sparse matrices with CVXPY?
-------------------------------------------
Yes, though you need to be careful.
SciPy sparse matrices do not support operator overloading to the extent needed by CVXPY.
(See `this Github issue <https://github.com/scipy/scipy/issues/4819>`_ for details.)
You can wrap a SciPy sparse matrix as a CVXPY constant, however, and then use it normally with CVXPY:

.. code:: python

  # Wrap the SciPy sparse matrix A as a CVXPY constant.
  A = Constant(A)
  # Use A normally in CVXPY expressions.
  expr = A*x

How do I constrain a CVXPY matrix expression to be positive semidefinite?
------------------------------------------------------------------------------
See :ref:`Advanced Features <advanced>`.

How do I create variables with special properties, such as boolean or symmetric variables?
-------------------------------------------------------------------------------------------
See :ref:`Advanced Features <advanced>`.

How do I create a variable that has multiple special properties, such as boolean and symmetric?
---------------------------------------------------------------------------------------------------
Create one variable with each desired property, and then set them all equal by adding equality constraints.
`CVXPY 1.0 <https://github.com/cvxgrp/cvxpy/issues/199>`_ will have a more elegant solution.

How do I create complex variables?
----------------------------------
You must represent complex variables using real variables,
as described in `this Github issue <https://github.com/cvxgrp/cvxpy/issues/191>`_.
We hope to add complex variables soon.

How do I create variables with more than 2 dimensions?
------------------------------------------------------
You must mimic the extra dimensions using a dict,
as described in `this Github issue <https://github.com/cvxgrp/cvxpy/issues/198>`_.

How does CVXPY work?
--------------------
The algorithms and data structures used by CVXPY are discussed in `this paper <http://arxiv.org/abs/1506.00760>`_.

How do I cite CVXPY?
--------------------
If you use CVXPY for published work, we encourage you to cite the software.
Use the following BibTeX citation:

::

    @misc{cvxpy,
      author       = {Steven Diamond and Eric Chu and Stephen Boyd},
      title        = {{CVXPY}: A {P}ython-Embedded Modeling Language for Convex Optimization, version 0.2},
      howpublished = {\url{http://cvxpy.org/}},
      month        = may,
      year         = 2014
    }
