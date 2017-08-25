.. _transforms-api:

Transforms
==========
Transforms provide additional ways of manipulating CVXPY objects
beyond the atomic functions.
While atomic functions operate only on expressions,
transforms may also take Problem, Objective, or Constraint objects as input.

.. automethod:: cvxpy.transforms.indicator

.. automethod:: cvxpy.transforms.linearize

.. automethod:: cvxpy.transforms.partial_optimize
