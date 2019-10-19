Problems
=======================
The :class:`~cvxpy.problems.problem.Problem` class is the entry point to
specifying and solving optimization problems. Each
:class:`cvxpy.problems.problem.Problem` instance encapsulates an optimization
problem, i.e., an objective and a set of constraints, and the
:meth:`~cvxpy.problems.problem.Problem.solve` method either solves the problem
encoded by the instance, returning the optimal value and setting variables
values to optimal points, or reports that the problem was in fact infeasible or
unbounded. You can construct a problem, solve it, and inspect both its value
and the values of its variables like so:

.. code:: python

    problem = Problem(Minimize(expression), constraints)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print "Optimal value: %s" % problem.value
    for variable in problem.variables():
        print "Variable %s: value %s" % (variable.name(), variable.value)

Problems are **immutable**, except through the specification of
:class:`~cvxpy.expressions.constants.parameter.Parameter` values. This means
that you **cannot** modify a problem's objective or constraints after you have
created it. If you find yourself wanting to add a constraint to an existing
problem, you should instead create a new problem using, for example, the
following idiom:

.. code:: python

    problem = Problem(Minimize(expression), constraints)
    problem = Problem(problem.objective, problem.constraints + new_constraints)

Most users need not know anything about the
:class:`~cvxpy.problems.problem.Problem` class except how to instantiate it,
how to solve problem instances (:meth:`~cvxpy.problems.problem.Problem.solve`),
and how to query the solver results
(:attr:`~cvxpy.problems.problem.Problem.status` and
:attr:`~cvxpy.problems.problem.Problem.value`).

Information about the size of a problem instance and statistics about the most
recent solve invocation are captured by the
:class:`~cvxpy.problems.problem.SizeMetrics` and
:class:`~cvxpy.problems.problem.SolverStats` classes, respectively, and can be
accessed via the :meth:`~cvxpy.problems.problem.Problem.size_metrics` and
:meth:`~cvxpy.problems.problem.Problem.solver_stats` properties of the
:class:`~cvxpy.problems.problem.Problem` class.

.. contents:: :local:


Minimize
--------
.. autoclass:: cvxpy.problems.objective.Minimize
    :members: is_dcp, is_dgp
    :undoc-members:
    :show-inheritance:

Maximize
--------
.. autoclass:: cvxpy.problems.objective.Maximize
    :members: is_dcp, is_dgp
    :undoc-members:
    :show-inheritance:

SizeMetrics
-----------
.. autoclass:: cvxpy.problems.problem.SizeMetrics
    :members:
    :undoc-members:
    :show-inheritance:

SolverStats
-----------
.. autoclass:: cvxpy.problems.problem.SolverStats
    :members:
    :undoc-members:
    :show-inheritance:

Problem
-------
.. autoclass:: cvxpy.problems.problem.Problem
    :members: value, status, objective, constraints, is_dcp, is_dgp, is_qp,
              variables, parameters, constants, atoms, size_metrics,
              solver_stats, solve, register_solve, get_problem_data,
              unpack_results,
    :undoc-members:
    :show-inheritance:

