Constraints
==========================
A constraint is an equality or inequality that restricts the domain of
an optimization problem. CVXPY has seven types of constraints: non-positive,
equality or zero, positive semidefinite, second-order cone, exponential
cone, 3-dimensional power cones, and N-dimensional power cones. The *vast*
majority of users will need only create constraints of the first three types.
Additionally, most users need not know anything more about constraints other
than how to create them. The constraint APIs do nonetheless provide methods that
advanced users may find useful; for example, some of the APIs allow you to
inspect dual variable values and residuals.

.. contents:: :local:

.. _constraint:

Constraint
---------------------------------

.. autoclass:: cvxpy.constraints.constraint.Constraint
    :members: value, violation, is_dcp
    :undoc-members:
    :show-inheritance:

.. _nonpos:

NonPos
---------------------------------

.. autoclass:: cvxpy.constraints.nonpos.NonPos
    :members: value, violation, is_dcp, shape, size, dual_value
    :undoc-members:
    :show-inheritance:

.. _zero:

Zero
-------------------------------

.. autoclass:: cvxpy.constraints.zero.Zero
    :members: value, violation, is_dcp
    :undoc-members:
    :show-inheritance:

.. _psd:

PSD
------------------------------

.. autoclass:: cvxpy.constraints.psd.PSD
    :members: value, violation, is_dcp
    :undoc-members:
    :show-inheritance:

SOC
----------------------------------------

.. autoclass:: cvxpy.constraints.second_order.SOC
    :members: value, violation, is_dcp
    :undoc-members:
    :show-inheritance:

.. _expcone:

ExpCone
--------------------------------------

.. autoclass:: cvxpy.constraints.exponential.ExpCone
    :members: value, violation, is_dcp
    :undoc-members:
    :show-inheritance:

PowCone3D
--------------------------------------

.. autoclass:: cvxpy.constraints.power.PowCone3D
    :members: value, violation, is_dcp
    :undoc-members:
    :show-inheritance:

PowConeND
--------------------------------------

.. autoclass:: cvxpy.constraints.power.PowConeND
    :members: value, violation, is_dcp
    :undoc-members:
    :show-inheritance:

FiniteSet
---------------------

.. autoclass:: cvxpy.constraints.finite_set.FiniteSet
    :members: is_dcp, size, shape, ineq_form, violation
    :undoc-members:
    :show-inheritance:

OpRelConeQuad
--------------------------------------

.. autoclass:: cvxpy.constraints.exponential.OpRelConeQuad
    :members: value, is_dcp
    :undoc-members:
    :show-inheritance:
