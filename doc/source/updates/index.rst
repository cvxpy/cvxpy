.. _updates:

Changes to CVXPY
================

This page details changes made to CVXPY over time, in reverse chronological order.
The latest release of CVXPY is version 1.1.

Recent patches
--------------

Changes in version 1.1.10
 - When NumPy 1.20 was released many users encountered errors in installing or importing
   CVXPY. Users would see errors like ``RuntimeError: module compiled
   against API version 0xe but this version of numpy is 0xd``. We changed our build files
   to avoid this problem, and it should be fixed as of CVXPY 1.1.10. For more information
   you can refer to this `GitHub issue <https://github.com/cvxpy/cvxpy/issues/1229>`_.

.. _changes118:

Changes in version 1.1.8
 - We have added support for 3-dimensional and N-dimensional power cone constraints. Although,
   we currently do not have any atoms that take advantage of this constraint. If you want
   you want to use this type of constraint in your model, you will need to instantiate
   ``PowCone3D`` and/or ``PowConeND`` objects manually. Dual variables are not yet implemented
   for ``PowConeND`` objects. At present, only SCS and MOSEK support power cone constraints.
 - We fixed a bug in our MOSEK interface that was introduced in version 1.1.6. The "unknown"
   status code was not being handled correctly, resulting in ValueErrors rather than SolverErrors.
   Users can now expect a SolverError when MOSEK returns an "unknown" status code (as was
   standard before).

.. _changes116:

Changes in version 1.1.6
 - The ECOS_BB solver (removed in 1.1.0) has been added back as an option. However ECOS_BB will not
   be called automatically; you must explicitly call ``prob.solve(solver='ECOS_BB')`` if you want to
   use this solver. Refer to our documentation on :ref:`mixed-integer models <mip>` for more information.
 - The MOSEK interface has been rewritten and now dualizes all continuous problems. Refer to :ref:`solver
   documentation <solveropts>` for technical reasons of why we do this, and how to manage MOSEK solver
   options in the off chance that this change made your solve times increase.



CVXPY 1.1
---------

Highlights
~~~~~~~~~~

:ref:`Disciplined parametrized programming <dpp>` or "DPP" is a ruleset for constructing parametrized problems in
CVXPY. Taking advantage of DPP can decrease the time it takes CVXPY to repeatedly canonicalize a parametrized problem.
DPP also provides the basis for differentiating the map from parameters to the solution of an optimization problem.

CVXPY provides an API where certain solvers can differentiate the map from the parameters of an
optimization problem to the optimal solution of that problem. The differentiation abilities are currently
only available when SCS is used as the solver.
This feature allows for more general sensitivity analysis than is possible when using dual variables alone. It also
provides the basis for `cvxpylayers <https://github.com/cvxgrp/cvxpylayers>`_.
See the :ref:`tutorial on derivatives <derivatives>`
and the `accompanying <https://web.stanford.edu/~boyd/papers/diff_cvxpy.html>`_
`papers <https://web.stanford.edu/~boyd/papers/diff_dgp.html>`_

Since version 0.4, CVXPY has used ``*`` to perform matrix multiplication. As of version 1.1,
this behavior is officially deprecated. All matrix multiplication should now be performed with
the python standard ``@`` operator. CVXPY will raise a warning if ``*`` is used when one of
the operands is not a scalar.

New atoms and transforms
~~~~~~~~~~~~~~~~~~~~~~~~

CVXPY has long provided abstractions ("atoms" and "transforms") which make it easier to specify
optimization problems in natural ways. The release of CVXPY 1.1 is accompanied by the following
new abstractions:

 - A "support function" transform for use in disciplined convex programming.
 - A "scalar product" atom, for appropriate use across all problem classes.
 - A "gmatmul" atom, which captures the DGP equivalent to matrix multiplication.
 - The atoms ``cp.max`` and ``cp.min`` have been extended for use in DQCP.
 - The python builtin ``sum`` is now allowed in DGP.

Breaking changes
~~~~~~~~~~~~~~~~

We no longer support Python 2 or Python 3.4.

This release drops the SuperSCS and ECOS_BB solvers.

Bugfixes
~~~~~~~~

CVXPY 1.1 has substantially improved support for recovering dual variables.
Advanced users should be able to recover dual variables to any conic constraint,
including exponential-cone and second-order-cone constraints.

This release resolves bugs in detecting when a problem falls into the category of
"disciplined quasiconvex programming" (DQCP).

Known issues
~~~~~~~~~~~~

DPP problems with many CVXPY Parameters can take a long time to compile.

Disciplined quasiconvex programming (DQCP) doesn't support DPP.

The XPRESS interface is currently not working.

Wishlist
~~~~~~~~

The following topics are (relatively) accessible to new contributors, and have
the potential to meaningfully improve CVXPY 1.1.

 - Extend more solver interfaces to allow differentiating the map from problem
   parameters to optimal solutions. In particular, extending the ECOS or CVXOPT
   interfaces. This may involve contributions to diffcp (see diffcp `GitHub issue
   31 <https://github.com/cvxgrp/diffcp/issues/31>`_).
 - Add :ref:`an interface <contrib_solver>` to an open-source mixed-integer nonlinear solver. CVXPY
   currently only supports commercial mixed-integer nonlinear solvers.
 - Help resolve any CVXPY GitHub issue with the label
   "`help wanted <https://github.com/cvxpy/cvxpy/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`_."

Anyone interested in making contributions should read the
:ref:`contributing guide<contributing>` before writing code.

CVXPY 1.0
---------

CVXPY 1.0 includes a major rewrite of the CVXPY internals, as well as a number of changes to the user interface. We first give an overview of the changes, before diving into the details.
We only cover changes that might be of interest to users.

We have created a script to convert code using CVXPY 0.4.11 into CVXPY 1.0, available `here <https://github.com/cvxpy/cvxpy/blob/1.0/cvxpy/utilities/cvxpy_upgrade.py>`_.

Overview
~~~~~~~~

* Disciplined geometric programming (DGP): Starting with version 1.0.11, CVXPY lets you formulate and solve log-log convex programs, which generalize both traditional geometric programs and generalized geometric programs. To get started with DGP, check out :ref:`the tutorial <dgp>` and consult the `accompanying paper <https://web.stanford.edu/~boyd/papers/dgp.html>`_.

* Reductions: CVXPY 1.0 uses a modular system of *reductions* to convert problems input by the user into the format required by the solver, which makes it easy to support new standard forms, such as quadratic programs, and more advanced user inputs, such as problems with complex variables. See :ref:`reductions-api` and the `accompanying paper <http://stanford.edu/~boyd/papers/cvxpy_rewriting.html>`_ for further details.

* Attributes: Variables and parameters now support a variety of attributes that describe their symbolic properties, such as nonnegative or symmetric. This unifies the treatment of symbolic properties for variables and parameters and replaces specialized variable classes such as ``Bool`` and ``Semidef``.

* NumPy compatibility: CVXPY's interface has been changed to resemble NumPy as closely as possible, including support for 0D and 1D arrays.

* Transforms: The new transform class provides additional ways of manipulating CVXPY objects, byond the atomic functions. While atomic functions operate only on expressions, transforms may also take Problem, Objective, or Constraint objects as input.



Reductions
~~~~~~~~~~

A reduction is a transformation 
from one problem to an equivalent problem. Two problems are equivalent
if a solution of one can be converted to a solution of the other with no
more than a moderate amount of effort. CVXPY uses reductions to rewrite
problems into forms that solvers will accept.
The practical benefit of the reduction based framework is that CVXPY 1.0 supports quadratic programs as a target solver standard form in addition to cone programs, with more standard forms on the way.
It also makes it easy to add generic problem transformations such as converting problems with complex variables into problems with only real variables.

Attributes
~~~~~~~~~~

Attributes describe the symbolic properties of variables and parameters and are specified as arguments to the constructor. For example, ``Variable(nonneg=True)`` creates a scalar variable constrained to be nonnegative.
Attributes replace the previous syntax of special variable classes like ``Bool`` for boolean variables and ``Semidef`` for symmetric positive semidefinite variables,
as well as specification of the sign for parameters (e.g., ``Parameter(sign='positive')``).
Concretely, write

* ``Variable(shape, boolean=True)`` instead of ``Bool(shape)``.
  
* ``Variable(shape, integer=True)`` instead of ``Int(shape)``.

* ``Variable((n, n), PSD=True)`` instead of ``Semidef(n)``.

* ``Variable((n, n), symmetric=True)`` instead of ``Symmetric(n)``.

* ``Variable(shape, nonneg=True)`` instead of ``NonNegative(shape)``.

* ``Parameter(shape, nonneg=True)`` instead of ``Parameter(shape, sign='positive')``.
 
* ``Parameter(shape, nonpos=True)`` instead of ``Parameter(shape, sign='negative')``.

See :ref:`attributes` for a complete list of supported attributes. More attributes will be added in the future.

NumPy Compatibility
~~~~~~~~~~~~~~~~~~~

The following interface changes have been made to make CVXPY more compatible with NumPy syntax:

* The ``value`` field of CVXPY expressions now returns NumPy ndarrays instead of NumPy matrices.

* The dimensions of CVXPY expressions are given by the ``shape`` field, while the ``size`` field gives the total number of entries. In CVXPY 0.4.11 and earlier, the ``size`` field gave the dimensions and the ``shape`` field did not exist.

* The dimensions of CVXPY expressions are no longer always 2D. 0D and 1D expressions are possible. We will add support for arbitrary ND expressions in the future. The number of dimensions is given by the ``ndim`` field.

* The shape argument of the ``Variable``, ``Parameter``, and ``reshape`` constructors must be a tuple. Instead of writing, ``Parameter(2, 3)`` to create a parameter of shape ``(2, 3)``, you must write ``Parameter((2, 3))``.

* Indexing and other operations can map 2D expressions down to 1D or 0D expressions. For example, if ``X`` has shape ``(3, 2)``, then ``X[:,0]`` has shape ``(3,)``. CVXPY behavior follows NumPy semantics in all cases, with the exception that broadcasting only works when one argument is 0D.

* Several CVXPY atoms have been renamed:

  * ``mul_elemwise`` to ``multiply``
  * ``max_entries`` to ``max``
  * ``sum_entries`` to ``sum``
  * ``max_elemwise`` to ``maximum``
  * ``min_elemwise`` to ``minimum``

* Due to the name changes, we now strongly recommend against importing CVXPY using the syntax ``from cvxpy import *``.

* The ``vstack`` and ``hstack`` atoms now take lists as input. For example, write ``vstack([x, y])`` instead of ``vstack(x, y)``.

Transforms
~~~~~~~~~~

Transforms provide additional ways of manipulating CVXPY objects
beyond the atomic functions.
For example, the ``indicator`` transform converts a list of constraints
into an expression representing the convex function that takes value 0 when
the constraints hold and :math:`\infty` when they are violated. See :ref:`transforms-api` for a full list of the new transforms.
