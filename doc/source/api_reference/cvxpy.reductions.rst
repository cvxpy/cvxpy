.. _reductions-api:

Reductions
==========

A :class:`~cvxpy.reductions.reduction.Reduction` is a transformation 
from one problem to an equivalent problem. Two problems are equivalent
if a solution of one can be converted to a solution of the other with no
more than a moderate amount of effort. CVXPY uses reductions to rewrite
problems into forms that solvers will accept.

Reductions allow CVXPY to simplify problems and target different
categories of solvers (quadratic program solvers and conic
solvers are two examples of solver categories). Appropriating terminology from
software compilers, we classify reductions as either middle-end reductions or
back-end reductions. A reduction that simplifies a source problem without
regard to the targeted solver is called a
:doc:`middle-end reduction <cvxpy.reductions.middle_end>`, whereas a reduction
that takes a source problem and converts it to a form acceptable to a category
of solvers is called a :doc:`back-end reduction <cvxpy.reductions.back_end>`.
Each solver (along with the mode in which it is invoked) is called a *back-end*
or *target*.

The majority of users do not need to know anything about CVXPY's reduction system.
Indeed, most users need not even know that reductions exist!

.. _reductions_disclaimer:

Disclaimer
~~~~~~~~~~

*CVXPY's reductions are not considered part of the public API. They can change
without notice in future releases.
We document them here for potential CVXPY contributors, for the curious, and for
those who are okay with building on an API that may not be available in future
versions of CVXPY.*


.. toctree::

    cvxpy.reductions.middle_end
    cvxpy.reductions.back_end

.. contents:: :local:

Solution
--------

.. autoclass:: cvxpy.reductions.solution.Solution
    :members:
    :show-inheritance:


Reduction
------------------------------------------

.. autoclass:: cvxpy.reductions.reduction.Reduction
    :members: __init__, accepts, reduce, retrieve, apply, invert
    :show-inheritance:

Chain
------------------------------------------

.. autoclass:: cvxpy.reductions.chain.Chain
    :members:
    :show-inheritance:

SolvingChain
------------------------------------------

.. autoclass:: cvxpy.reductions.solvers.solving_chain.SolvingChain
    :members:
    :show-inheritance:
