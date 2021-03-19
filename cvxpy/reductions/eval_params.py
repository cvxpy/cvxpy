from cvxpy.error import ParameterError
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy import problems
from cvxpy.reductions.reduction import Reduction


def replace_params_with_consts(expr):
    if isinstance(expr, list):
        return [replace_params_with_consts(elem) for elem in expr]
    elif len(expr.parameters()) == 0:
        return expr
    elif isinstance(expr, Parameter):
        if expr.value is None:
            raise ParameterError("Problem contains unspecified parameters.")
        return Constant(expr.value)
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(replace_params_with_consts(arg))
        return expr.copy(new_args)


class EvalParams(Reduction):
    """Replaces symbolic parameters with their constant values."""

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        """Replace parameters with constant values.

        Parameters
        ----------
        problem : Problem
            The problem whose parameters should be evaluated.

        Returns
        -------
        Problem
            A new problem where the parameters have been converted to constants.

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
