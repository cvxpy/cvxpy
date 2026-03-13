.. _cvxpygen-explicit:

Explicitly Solving Problems
============================

For quadratic programs in which 
the coefficients of the linear objective terms and the righthand side of the
constraints are affine functions of a parameter,
the solution is a piecewise affine function of the parameter.

The number of (polyhedral) regions in the solution map
can grow exponentially in problem size (specifically, the number of inequality constraints),
but when the number of regions is moderate, a so-called 
explicit solver is practical.
Such a solver computes the coefficients of the affine
functions and the linear inequalities defining the polyhedral regions 
offline; to solve a problem instance online it simply evaluates 
this explicit solution map.
Potential advantages of an explicit solver over a more general purpose
iterative solver can include transparency, interpretability, reliability,
and speed.

Generating Explicit Solvers
----------------------------

CVXPYgen can generate such explicit solvers.
To enable this feature, set ``solver='explicit'`` when generating code.
By default, only the primal solution is computed. To also compute the dual
solution, pass ``solver_opts={'dual': True}``.
You can choose to store the explicit solution in half precision (instead of single
precision), by setting ``'fp16': True`` in ``solver_opts``.
Limits on parameters are encouraged and can be represented as standard CVXPY constraints.
As of now, we support simple bounds of the form ``[l <= p, p <= u]`` where
``p`` is a ``cp.Parameter()`` and ``l`` and ``u`` are constants.

Configuration Options
---------------------

See our `manuscript <https://stanford.edu/~boyd/papers/cvxpygen_mpqp.html>`_ for more details.
You can control the maximum number of floating point numbers in the explicit solution
via ``'max_floats'`` (default is ``1e6``) and the maximum number of regions
via ``'max_regions'`` (default is ``500``) in the ``solver_opts`` dict.

Implementation
--------------

CVXPYgen uses `PDAQP <https://github.com/darnstrom/pdaqp>`_ to construct explicit solutions.
