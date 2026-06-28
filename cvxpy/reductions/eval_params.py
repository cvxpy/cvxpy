from cvxpy import problems
from cvxpy.error import ParameterError
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.reductions.reduction import Reduction


def replace_params_with_consts(expr):
    """Fold variable-free *composite* parametric subtrees to their values.

    Bare ``Parameter`` leaves are kept symbolic so that variable-coupled
    parameter usage (e.g. ``A @ x``, ``square(p * x)``) reaches the DIFFENGINE
    backend, which re-evaluates it on each solve. Only *composite* subtrees that
    are parametric and contain no optimization variables (e.g. ``log_det(P)``,
    ``A @ B``) are folded to constants -- these are the terms that would
    otherwise needlessly force cones or break the affine-in-parameter structure.
    """
    if isinstance(expr, list):
        return [replace_params_with_consts(elem) for elem in expr]
    elif len(expr.parameters()) == 0:
        return expr
    elif isinstance(expr, Parameter):
        # Keep bare parameters symbolic.
        return expr
    elif len(expr.variables()) == 0:
        # Variable-free parametric composite: fold the whole subtree to its
        # numeric value (it only contributes a constant offset to the program).
        if expr.value is None:
            raise ParameterError("Problem contains unspecified parameters.")
        return Constant(expr.value)
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(replace_params_with_consts(arg))
        return expr.copy(new_args)


class EvalParams(Reduction):
    """Folds variable-free composite parametric subtrees to constants.

    Variable-free parametric composites (e.g. ``log_det(P)``, ``A @ B``) are
    replaced by their numeric values, which avoids needlessly forcing cones.
    Bare parameters are kept symbolic so that variable-coupled parameter usage
    flows to the DIFFENGINE backend, which re-evaluates it on each solve. See
    :func:`replace_params_with_consts`.
    """

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        """Fold variable-free composite parametric subtrees.

        Parameters
        ----------
        problem : Problem
            The problem whose parameters should be evaluated.

        Returns
        -------
        Problem
            A new problem where the variable-free composite parametric subtrees
            have been converted to constants.

        Raises
        ------
        ParameterError
            If the ``problem`` has unspecified parameters (i.e., a parameter
            whose value is None).
        """
        # Do not instantiate a new objective if it does not contain
        # parameters.
        if len(problem.objective.parameters()) > 0:
            obj_expr = replace_params_with_consts(problem.objective.expr)
            objective = type(problem.objective)(obj_expr)
        else:
            objective = problem.objective

        constraints = []
        for c in problem.constraints:
            args = []
            for arg in c.args:
                args.append(replace_params_with_consts(arg))
            # Do not instantiate a new constraint object if it did not
            # contain parameters.
            if all(id(new) == id(old) for new, old in zip(args, c.args)):
                constraints.append(c)
            # Otherwise, create a copy of the constraint.
            else:
                data = c.get_data()
                if data is not None:
                    constraints.append(type(c)(*(args + data)))
                else:
                    constraints.append(type(c)(*args))
        return problems.problem.Problem(objective, constraints), []

    def invert(self, solution, inverse_data):
        """Returns a solution to the original problem given the inverse_data.
        """
        return solution
