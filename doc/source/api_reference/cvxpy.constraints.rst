Constraints
==========================
A constraint is an equality or inequality that restricts the domain of
an optimization problem. CVXPY has five types of constraints: non-positive,
equality or zero, positive semidefinite, second-order cone, and exponential
cone. The *vast* majority of users will need only create constraints of the
first three types, and these constraints are most naturally constructed
using the ``>=`` / ``<=`` (for non-negative and non-positive constraints),
``==`` (for equality constraints), and ``>>`` / ``<<`` (for
positive-semidefinite and negative-semidefinite constraints) operators of the
``Expression`` class. The constraint classes do, however, expose some methods
that you may find useful, and as such we include their documentation below.

NonPos
---------------------------------

.. autoclass:: cvxpy.constraints.nonpos.NonPos
    :members:
    :undoc-members:
    :show-inheritance:

Zero
-------------------------------

.. autoclass:: cvxpy.constraints.zero.Zero
    :members:
    :undoc-members:
    :show-inheritance:

SOC
----------------------------------------

.. autoclass:: cvxpy.constraints.second_order.SOC
    :members:
    :undoc-members:
    :show-inheritance:

PSD
------------------------------

.. autoclass:: cvxpy.constraints.psd.PSD
    :members:
    :undoc-members:
    :show-inheritance:

ExpCone
--------------------------------------

.. autoclass:: cvxpy.constraints.exponential.ExpCone
    :members:
    :undoc-members:
    :show-inheritance:
