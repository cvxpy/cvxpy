Back-End Reductions
====================

The reductions listed here are specific to the choice of back end, i.e.,
solver. Currently, we support two types of back ends: conic solvers and
quadratic program solvers.  When a problem is solved through the
:meth:`~cxvpy.problems.problem.Problem.solve` method, CVXPY attempts to find
the best back end for your problem. The
:class:`~cvxpy.reductions.dcp2cone.Dcp2Cone` reduction converts DCP-compliant
problems into conic form, while the
:class:`~cvxpy.reductions.qp2symbolic_qp.Qp2SymbolicQp` converts problems with
quadratic, piecewise affine objectives, affine equality constraints, and
piecewise-linear inequality constraints into a form that is closer to what is
accepted by solvers. The problems output by both reductions must be passed
through another sequence of reductions, not documented here, before they are
ready for to be solved.

.. contents:: :local:

Dcp2Cone
------------------------------------------

.. autoclass:: cvxpy.reductions.dcp2cone.dcp2cone.Dcp2Cone
    :members:
    :show-inheritance:


Qp2SymbolicQp
------------------------------------------

.. autoclass:: cvxpy.reductions.qp2quad_form.qp2symbolic_qp.Qp2SymbolicQp
    :members:
    :show-inheritance:
