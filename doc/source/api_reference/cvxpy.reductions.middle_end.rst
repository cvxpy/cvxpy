Middle-End Reductions
====================

The reductions listed here are not specific to a type of solver.
They can be applied regardless of whether you wish to target, for example,
a quadratic program solver or a conic solver.

Please see `our disclaimer <reductions_disclaimer>`_ about the
Reductions API before using these directly in your code.

.. contents:: :local:

Complex2Real
------------------------------------------

.. autoclass:: cvxpy.reductions.complex2real.complex2real.Complex2Real
    :members:
    :show-inheritance:

CvxAttr2Constr
------------------------------------------

.. autoclass:: cvxpy.reductions.cvx_attr2constr.CvxAttr2Constr
    :members:
    :show-inheritance:

Dgp2Dcp
------------------------------------------

.. autoclass:: cvxpy.reductions.dgp2dcp.dgp2dcp.Dgp2Dcp
    :members:
    :show-inheritance:

EvalParams
------------------------------------------

.. autoclass:: cvxpy.reductions.eval_params.EvalParams
    :members:
    :show-inheritance:

FlipObjective
------------------------------------------

.. autoclass:: cvxpy.reductions.flip_objective.FlipObjective
    :members:
    :show-inheritance:
