.. _transforms-api:

Transforms
==========
Transforms provide additional ways of manipulating CVXPY objects
beyond the atomic functions.
While atomic functions operate only on expressions,
transforms may also take Problem, Objective, or Constraint objects as input.


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

.. automethod:: cvxpy.transforms.partial_optimize
