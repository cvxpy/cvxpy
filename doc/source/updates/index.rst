.. _updates:

Changes to CVXPY
================

This page details changes made to CVXPY over time, in reverse chronological order.
CVXPY's project maintainers currently provide support for CVXPY 1.7.

CVXPY 1.7
---------

This release is consistent with our semantic versioning guarantee. It
comes packed with many new features, bug fixes, and performance improvements.
This version of CVXPY supports Python 3.10 through 3.13. While working on the next release,
we continue to officially support CVXPY 1.6.

New GPU solvers
~~~~~~~~~~~~~~~

CVXPY begins supporting GPU solvers in this release. 
The following solvers are supported:
- `MPAX <https://github.com/MIT-Lu-Lab/MPAX>`_
- `cuOpt <https://github.com/NVIDIA/cuopt>`_
- `CuClarabel <https://github.com/cvxgrp/CuClarabel>`_
- `SCS <https://github.com/bodono/scs-python/pull/136>`_

MPAX runs on a GPU device specified by the JAX environment. MPAX, cuOpt, and CuClarabel
are new solver interfaces that can be used with CVXPY. SCS has a new backend based on
`cuDSS <https://developer.nvidia.com/cudss>`_ that can be used through the existing SCS interface.

Sparse array support
~~~~~~~~~~~~~~~~~~~~

SciPy is deprecating the sparse matrix API in favor of sparse arrays. See the 
migration guide `here <https://docs.scipy.org/doc/scipy/reference/sparse.migration_to_sparray.html#migration-to-sparray>`_.
CVXPY 1.7 supports the new sparse array API, both internally and externally, but continues to support the sparse matrix API
for backwards compatibility. However, sparse matrix inputs will be converted to sparse arrays after CVXPY's canonicalization.

cvxpy-base standard distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



New features
~~~~~~~~~~~~
- New feature: :ref:`Multiple attributes <multiple-attributes>` for variables and parameters
- New `QOCO <https://qoco-org.github.io/qoco/>`_ solver interface
- New atom: :ref:`broadcast_to <broadcast_to>`
- New atom: :ref:`transpose(expr, axes) <transpose>`
- New atom: :ref:`swapaxes <swapaxes>`
- New atom: :ref:`moveaxis <moveaxis>`
- New atom: :ref:`permute_dims <permute_dims>`
- Add warm-start support for :ref:`HiGHS <HiGHS>` (LP and MIP)
- Add warm-start support for :ref:`PIQP <PIQP>`

CVXPY 1.6
---------

This release is consistent with our semantic versioning guarantee. It
comes packed with many new features, bug fixes, and performance improvements.
This version of CVXPY supports Python 3.9 through 3.13. While working on the next release,
we continue to officially support CVXPY 1.5.

Default reshape order warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CVXPY's default order for array manipulation atoms such as reshape, vec and flatten,
is Fortran ('F'). In this release CVXPY raises a warning when no explicit order is specified.

In version 1.7, we plan to raise an error if the order is not specified.
Finally, in version 1.8, we will switch the default order from ('F') to ('C') to
match NumPy's behavior.

Dropping ECOS dependency
~~~~~~~~~~~~~~~~~~~~~~~~

In version 1.5, we changed our default solver from ECOS to Clarabel and announced that we would be
removing ECOS as a dependency in 1.6. Despite some regressions in certain DQCP tests, we are
moving forward with dropping ECOS in this release. If you are experiencing any issues with Clarabel
we encourage you to try using SCS or add ECOS as a dependency to your project.

New features
~~~~~~~~~~~~

- Added Python 3.13 support and dropped Python 3.8 support
- New HiGHS solver interface
- New atom: :ref:`cvar <cvar>`
- New atom: :ref:`cumprod <cumprod>`
- New atom: :ref:`quantum_rel_entr <quantum_rel_entr>`
- New atom: :ref:`quantum_cond_entr <quantum_cond_entr>`
- New atom: :ref:`concatenate <concatenate>`
- Support for :ref:`N-dimensional variables <n-dimensional>` and expressions for the following operations:
    * axis atoms like min, max and sum
    * indexing
    * elementwise operations
- :ref:`Sparsity attribute <sparsity>` for variables
- New website and documentation theme based on `Sphinx Immaterial <https://jbms.github.io/sphinx-immaterial/>`_
- Ability to pass multiple solvers as argument to ``.solve()``
- Performance improvement for ``sum_largest`` and ``cumsum``
- Performance improvement for integer and boolean variables
- Improving string representation of special index


CVXPY 1.5
---------

This release is consistent with our semantic versioning guarantee. It
comes packed with many new features, bug fixes, and performance
improvements. This version of CVXPY supports Python 3.8 through 3.12,
While working on the next release, we continue to officially support
CVXPY 1.5 and 1.4.

This release may **not** be compatible with NumPy 2.0.


ECOS deprecation
~~~~~~~~~~~~~~~~

CVXPY has used ECOS as the default solver for many years; however, it
has known issues with performance and numerical stability in edge cases.
Recently, a new solver, Clarabel, that improves the algorithm and
implementation of ECOS has been under development.

In this release, CVXPY uses Clarabel instead of ECOS for all
categories of problems where ECOS was previously the default.

In 1.6, we plan to no longer install ECOS as a CVXPY dependency.
We have no plans to remove support for calling ECOS as a solver.

We encourage you to try and use Clarabel instead, but if you're
dependent on ECOS's exact behavior please explicitly specify it as a
solver and as a dependency for your project.

New features
~~~~~~~~~~~~

- Major updates to the documentation, adding a number of new sections to the 
    User Guide and breaking up the monolithic Advanced features page
- Added `.curvatures` containing all curvatures an expression is compatible with
- Variable bounds can be specified with `cp.Variable(bound=(lower, upper))`
    and are directly passed to the solver when helpful. `lower` and `upper` can
    be either a NumPy array or floating point number.
- Constants can be named by writing `cp.Constant(name='...')`
- Added a new atom, `vdot`, that has the same behavior as `scalar_product`
- CVXPY runs in the next PyOdide release via wasm
- Added or-tools 9.9 support
- Major rewrite to the PDLP interface
- Dropped MOSEK <= 9 support and upgraded the MOSEK integration code

CVXPY 1.4
---------

This release is consistent with our semantic versioning guarantee. It
comes packed with many new features, bug fixes, and performance
improvements. This version of CVXPY supports Python 3.8 through 3.12,
and is our first release that supports Python 3.12. While working on the
next release, we continue to officially support CVXPY 1.3 and 1.4.

New features
~~~~~~~~~~~~
-  New atom: :ref:`convolve <convolve>`
-  New atom: :ref:`mean <mean>`
-  New atom: :ref:`outer <outer>`
-  New atom: :ref:`ptp <ptp>`
-  New atom: :ref:`std <std>`
-  New atom: :ref:`var <var>`
-  New atom: :ref:`vec_to_upper_tri <vec-to-upper-tri>`
-  Adds methods to CVXPY expressions that are found on NumPy ndarrays such as ``.sum()``, ``.max()``, and ``.mean()``
-  New solver interface: ``PIQP``
-  Adds SDP support to the Clarabel interface
-  Added support for OR-Tools 9.7
-  Removed support for OR-Tools 9.4
-  ``PowerConeND`` now supports extracting its dual variables
-  ``reshape`` now supports using ``-1`` as a dimension, with the same
   meaning it has in NumPy
-  Indexing CVXPY expressions with floats now raises an appropriate
   error
-  Clearer error messages for a number of common errors
-  The :ref:`perspective <perspective>` atom now supports ``s=0``
-  Performance improvements in the SCIPY backend
-  Performance improvements in canonicalizing parameterized QPs 
-  Performance improvements for quadratic forms with sparse matrices
-  Greater support for static typing

ECOS deprecation
~~~~~~~~~~~~~~~~

CVXPY has used ECOS as the default solver for many years; however, it
has known issues with performance and numerical stability in edge cases.
Recently, a new solver, Clarabel, that improves the algorithm and
implementation of ECOS has been under development.

In 1.5, CVXPY plans to start using Clarabel instead of ECOS by default for some
categories of problems.
In 1.6, we plan to no longer install ECOS as a CVXPY dependency.
We have no plans to remove support for calling ECOS as a solver.
As part of this transition, in 1.4 CVXPY will raise a warning whenever
ECOS is called by default.
We encourage you to try and use Clarabel instead, but if you're
dependent on ECOS's exact behavior please explicitly specify it as a
solver.

``conv`` deprecation
~~~~~~~~~~~~~~~~~~~~

The CVXPY atom ``conv`` is inconsistent with NumPy's convolve functions.
We are deprecating it, but have no plans to remove it in the short term.
We encourage all users to use the CVXPY atom ``convolve`` instead.

``NonPos`` deprecation
~~~~~~~~~~~~~~~~~~~~~~

The ``NonPos`` cone uses the opposite dual variable sign convention as
the rest of the CVXPY cones and a constraint of ``NonPos(expr)`` is the
same as a constraint on ``NonNeg(-expr)``. We are deprecating
``NonPos``, but have no plans to remove it in the short term. We
encourage users to switch to using ``NonNeg``.

CVXPY 1.3
---------
CVXPY 1.3 brings many new features, bug fixes, and performance improvements. It introduces a new
:ref:`SciPy-based backend <canonicalization-backends>` and formalizes the public API of CVXPY as everything that is
importable directly from the ``cvxpy`` namespace.
We plan to introduce a ``cvxpy.experimental`` namespace for features in development where
the API has not yet been fixed. It is explicitly not a part of our API whether atoms are implemented by functions
or classes, e.g. we do not consider replacing ``cvxpy.power``, which is currently a class, with a function to be a
breaking change or replacing ``cp.quad_form`` which is a function to become a class to be a breaking change.
Code of the form ``cvxpy.power(a, b)`` is guaranteed to remain working.

Constraints and atoms
~~~~~~~~~~~~~~~~~~~~~
- :ref:`FiniteSet <finite_set>`
- :ref:`RelEntrConeQuad <rel_entr_cone_quad>`
- :ref:`OpRelEntrConeQuad <op_rel_entr_cone_quad>`
- :ref:`dotsort(X,W) <dotsort>`
- :ref:`tr_inv(X) <tr_inv>`
- :ref:`von_neumann_entr(X) <von-neumann-entr>`
- :ref:`perspective(f(x),s) <perspective>`

Solver interfaces
~~~~~~~~~~~~~~~~~
- :ref:`New interfaces <solvers>`: COPT, SDPA, Clarabel, and proxqp

General system improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Support for native quadratic forms (``x.T @ P @ x``)
- The new OpRelEntrConeQuad constraint class is the first major piece of our effort to improve support for quantum
  information modeling (`GSOC project <https://github.com/cvxpy/org/blob/main/GSoC2022/aryamanjeendgar/final_report.pdf>`_)
- Continuous performance benchmarking (`GSOC project <https://github.com/cvxpy/org/blob/main/GSoC2022/parthb83/final_report.md>`_)


CVXPY 1.2
---------
We're taking a big step toward `semantic versioning <https://semver.org/>`_!
Our new versioning policy will be to increment the minor version number (the "x" in "CVXPY 1.x.y")
whenever we introduce new features.
The patch number (the "y" in "CVXPY 1.x.y") will only be incremented for bugfixes.
We'll support multiple minor releases of CVXPY at any given time.
API-breaking changes will require incrementing the major version number (i.e., moving to CVXPY 2.x.y).

This versioning policy is very different from what we've done in the past.
Many new features were added *after* CVXPY 1.1.0 but *before* CVXPY 1.2.0.
These features accumulated over the course of CVXPY 1.1.1 and 1.1.18.
We review those features and the new features in CVXPY 1.2.0 below.

Constraints and atoms
~~~~~~~~~~~~~~~~~~~~~
* 1.2.0: added atoms for `partial trace <https://en.wikipedia.org/wiki/Partial_trace>`_ and partial transpose,
  which are important linear operators in quantum information
* 1.2.0: updated ``kron`` so that either argument in ``kron(A, B)`` can be a non-constant affine Expression,
  provided the other argument is constant. We previously required that ``A`` was constant.
* 1.2.0: added ``xexp``: an atom that implements :math:`\texttt{xexp}(x) = x e^{x}`.
* 1.1.14: added ``loggamma``: an atom which approximates the log of the gamma function
* 1.1.14: added ``rel_entr``: an atom with the same semantics as the SciPy's "rel_entr"
* 1.1.8: added ``log_normcdf``: an atom that approximates the log of the Gaussian distribution's CDF
* 1.1.8: added power cone constraints

Solver interfaces
~~~~~~~~~~~~~~~~~
* 1.2.0: support PDLP and GLOP, via OR-Tools
* 1.1.17: support for SCS 3.0
* 1.1.14: support for HiGHS (and other LP solvers that come with SciPy)
* 1.1.12: ECOS, ECOS_BB, and SCS report solver statistics
* 1.1.12: support warm-start with GUROBI
* 1.1.8: added a mechanism for users to create solver interfaces without modifying CVXPY source code
* 1.1.6: rewrote the MOSEK interface; it now dualizes all continuous problems
* 1.1.4: support for FICO XPRESS
* 1.1.2: support for SCIP
* 1.1.2: users can provide their own implementation of a KKT solver for use with CVXOPT

General system improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* 1.1.18: A problem status "infeasible or unbounded", for use by specific solvers in rare situations
* 1.1.11: verbose logging
* 1.1.11: several improvements to CVXPY's  C++ backend rewriting system, "cvxcore."
  In particular, CVXPY can now be compiled from source with openmp enabled, which allows
  canonicalization to take advantage of multithreading.
* 1.1.6: a "Dualize" reduction

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

CVXPY 1.1.0 drops the SuperSCS and ECOS_BB solvers.

.. note::

	We added ECOS_BB back in version 1.1.6. Starting with
	CVXPY 1.2.0, any backwards-incompatible change like removing a
	solver interface will require incrementing CVXPY's major version
	number (e.g., moving from series 1.X to 2.X).

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

The XPRESS interface is currently not working. (Fixed in CVXPY 1.1.4.)


Notable patches since CVXPY 1.1.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 1.1.10
 - When NumPy 1.20 was released many users encountered errors in installing or importing
   CVXPY. Users would see errors like ``RuntimeError: module compiled
   against API version 0xe but this version of numpy is 0xd``. We changed our build files
   to avoid this problem, and it should be fixed as of CVXPY 1.1.10. For more information
   you can refer to this `GitHub issue <https://github.com/cvxpy/cvxpy/issues/1229>`_.

.. _changes118:

Version 1.1.8
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

Version 1.1.6
 - The ECOS_BB solver (removed in 1.1.0) has been added back as an option. However ECOS_BB will not
   be called automatically; you must explicitly call ``prob.solve(solver='ECOS_BB')`` if you want to
   use this solver. Refer to our documentation on :ref:`mixed-integer models <mip>` for more information.
 - The MOSEK interface has been rewritten and now dualizes all continuous problems. Refer to :ref:`solver
   documentation <solveropts>` for technical reasons of why we do this, and how to manage MOSEK solver
   options in the off chance that this change made your solve times increase.


CVXPY 1.0
---------

CVXPY 1.0 includes a major rewrite of the CVXPY internals, as well as a number of changes to the user interface. We first give an overview of the changes, before diving into the details.
We only cover changes that might be of interest to users.

We have created a script to convert code using CVXPY 0.4.11 into CVXPY 1.0, available `here <https://github.com/cvxpy/cvxpy/blob/1.0/cvxpy/utilities/cvxpy_upgrade.py>`_.

Overview
~~~~~~~~

* Disciplined geometric programming (DGP): Starting with version 1.0.11, CVXPY lets you formulate and solve log-log convex programs, which generalize both traditional geometric programs and generalized geometric programs. To get started with DGP, check out :ref:`the tutorial <dgp>` and consult the `accompanying paper <https://web.stanford.edu/~boyd/papers/dgp.html>`_.

* Reductions: CVXPY 1.0 uses a modular system of *reductions* to convert problems input by the user into the format required by the solver, which makes it easy to support new standard forms, such as quadratic programs, and more advanced user inputs, such as problems with complex variables. See :ref:`reductions-api` and the `accompanying paper <https://stanford.edu/~boyd/papers/cvxpy_rewriting.html>`_ for further details.

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
