.. _reductions-api:

Reductions
==========

A :class:`~cvxpy.reductions.reduction.Reduction` is a transformation 
from one problem to an equivalent problem. Two problems are equivalent
if a solution of one can be converted to a solution of the other with no
more than a moderate amount of effort. CVXPY uses reductions to rewrite
problems into forms that solvers will accept.

.. _reductions_disclaimer:

Disclaimer
~~~~~~~~~~

The majority of users do not need to know anything about CVXPY's reduction system.
Indeed, most users need not even know that reductions exist!

In order to make sure that we have flexibility to improve CVXPY's speed
and capabilities while preserving backwards-compatibility, we have
determined that the reduction system will not be considered part of CVXPY's
public API. As such, aspects of this system can change without notice in future releases.

We provide this documentation for CVXPY contributors, for the curious, and for
those who are okay with building on an API that may not be available in future
versions of CVXPY.
Please contact us if you have an interesting application which requires direct
access to the Reduction system and if you care about forward compatibility with
future versions of CVXPY.

Types of Reductions
~~~~~~~~~~~~~~~~~~~

Reductions allow CVXPY to support different problem classes such as convex
and log-log convex programming. They also help CVXPY target different categories
of solvers, such as quadratic programming solvers and conic solvers.

Appropriating terminology from software compilers, we classify reductions as
either middle-end reductions or back-end reductions. A reduction that simplifies
a source problem without regard to the targeted solver is called a
:doc:`middle-end reduction <cvxpy.reductions.middle_end>`, whereas a reduction
that takes a source problem and converts it to a form acceptable to a category
of solvers is called a :doc:`back-end reduction <cvxpy.reductions.back_end>`.
Each solver (along with the mode in which it is invoked) is called a *back-end*
or *target*.

Here's a breakdown of CVXPY's reductions:

.. toctree::

    Middle-End <cvxpy.reductions.middle_end>
    Back-End <cvxpy.reductions.back_end>

Core classes
~~~~~~~~~~~~

Reductions exchange data and operate by way of the following classes.
Other data structures can be used, such as dicts keyed by strings, but
these are the main classes which define the Reduction system.


Solution
--------

.. autoclass:: cvxpy.reductions.solution.Solution
    :members:


Reduction
------------------------------------------

.. autoclass:: cvxpy.reductions.reduction.Reduction
    :members: __init__, accepts, reduce, retrieve, apply, invert

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
