.. _updates:

What's New in 1.0
=======================

CVXPY 1.0 includes a major rewrite of the CVXPY internals, as well as a number of changes to the user interface. We first give an overview of the changes, before diving into the details.
We only cover changes that might be of interest to users.

We have created a script to convert code using CVXPY 0.4.11 into CVXPY 1.0, available `here <https://github.com/cvxgrp/cvxpy/blob/1.0/cvxpy/utilities/cvxpy_upgrade.py>`_.

Overview
--------

* Disciplined geometric programming (DGP): Starting with version 1.0.11, CVXPY lets you formulate and solve log-log convex programs, which generalize both traditional geometric programs and generalized geometric programs. To get started with DGP, check out :ref:`the tutorial <dgp>` and consult the `accompanying paper <https://web.stanford.edu/~boyd/papers/dgp.html>`_.

* Reductions: CVXPY 1.0 uses a modular system of *reductions* to convert problems input by the user into the format required by the solver, which makes it easy to support new standard forms, such as quadratic programs, and more advanced user inputs, such as problems with complex variables. See :ref:`reductions-api` and the `accompanying paper <http://stanford.edu/~boyd/papers/cvxpy_rewriting.html>`_ for further details.



* Attributes: Variables and parameters now support a variety of attributes that describe their symbolic properties, such as nonnegative or symmetric. This unifies the treatment of symbolic properties for variables and parameters and replaces specialized variable classes such as ``Bool`` and ``Semidef``.

* NumPy compatibility: CVXPY's interface has been changed to resemble NumPy as closely as possible, including support for 0D and 1D arrays.

* Transforms: The new transform class provides additional ways of manipulating CVXPY objects, byond the atomic functions. While atomic functions operate only on expressions, transforms may also take Problem, Objective, or Constraint objects as input.



Reductions
----------

A reduction is a transformation 
from one problem to an equivalent problem. Two problems are equivalent
if a solution of one can be converted to a solution of the other with no
more than a moderate amount of effort. CVXPY uses reductions to rewrite
problems into forms that solvers will accept.
The practical benefit of the reduction based framework is that CVXPY 1.0 supports quadratic programs as a target solver standard form in addition to cone programs, with more standard forms on the way.
It also makes it easy to add generic problem transformations such as converting problems with complex variables into problems with only real variables.

Attributes
----------

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
-------------------

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
----------

Transforms provide additional ways of manipulating CVXPY objects
beyond the atomic functions.
For example, the ``indicator`` transform converts a list of constraints
into an expression representing the convex function that takes value 0 when
the constraints hold and :math:`\infty` when they are violated. See :ref:`transforms-api` for a full list of the new transforms.
