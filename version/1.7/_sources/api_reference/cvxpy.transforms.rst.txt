.. _transforms-api:

Transforms
==========
Transforms provide additional ways of manipulating CVXPY objects
beyond the atomic functions.
While atomic functions operate only on expressions,
transforms may also take Problem, Objective, or Constraint objects as input.
Transforms do not need to conform to any specific API.

SuppFunc
--------

The *SuppFunc* transform accepts an implicit representation of a convex set in terms
of some CVXPY Variable and returns a function handle representing the convex set's
support function. When the function handle is evaluated it returns a `SuppFuncAtom object <suppfuncatom>`_.
Such objects can be used like any other CVXPY Expression for purposes of convex optimization modeling.

.. autoclass:: cvxpy.transforms.suppfunc.SuppFunc
    :members: __call__


Scalarize
---------

The *scalarize* transforms convert a list of objectives into a single objective,
for example a weighted sum. All scalarizations are monotone in each objective, which means that optimizing over the scalarized objective always returns a Pareto-optimal point with respect to the original list of objectives.
Moreover, all points on the Pareto curve except for boundary points can be attained given some weighting of the objectives.

.. automethod:: cvxpy.transforms.scalarize.weighted_sum

.. automethod:: cvxpy.transforms.scalarize.max

.. automethod:: cvxpy.transforms.scalarize.log_sum_exp

.. automethod:: cvxpy.transforms.scalarize.targets_and_priorities

Other
-----

Here we list other available transforms.

.. autoclass:: cvxpy.transforms.indicator

.. automethod:: cvxpy.transforms.linearize

.. automethod:: cvxpy.transforms.partial_optimize.partial_optimize
