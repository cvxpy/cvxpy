import numpy as np

import cvxpy.settings as s
from cvxpy.atoms import reshape
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
from cvxpy.utilities.coeff_extractor import CoeffExtractor
from cvxpy.reductions.inverse_data import InverseData


class ConeMatrixStuffing(Reduction):
    """Construct matrices for linear cone problems.

    Linear cone problems are assumed to have a linear objective and cone
    constraints which may have zero or more arguments, all of which must be
    affine.

    minimize   c'x
    subject to cone_constr1(A_1*x + b_1, ...)
               ...
               cone_constrK(A_i*x + b_i, ...)
    """

    def accepts(self, problem):
        return (
            problem.is_dcp() and
            problem.objective.args[0].is_affine() and
            all([arg.is_affine() for c in problem.constraints for arg in c.args])
        )

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.
        """
        objective = problem.objective
        constraints = problem.constraints

        inverse_data = InverseData(problem)
        N = inverse_data.x_length

        extractor = CoeffExtractor(inverse_data)

        # Extract the coefficients
        C, R = extractor.get_coeffs(objective.args[0])
        c = np.asarray(C.todense()).flatten()
        r = R[0]
        x = Variable(N)
        if type(objective) == Minimize:
            new_obj = c.T*x + r
        else:
            new_obj = (-c).T*x + -r
        # Form the constraints
        new_cons = []
        con_map = {}
        for con in constraints:
            arg_list = []
            for arg in con.args:
                A, b = extractor.get_coeffs(arg)
                arg_list.append(reshape(A*x + b, arg.shape))
            new_cons.append(type(con)(*arg_list))
            con_map[con.id] = new_cons[-1].id

        # Map of old constraint id to new constraint id.
        inverse_data.con_map = con_map
        inverse_data.minimize = type(problem.objective) == Minimize
        new_prob = Problem(Minimize(new_obj), new_cons)
        return (new_prob, inverse_data)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        var_map = inverse_data.var_offsets
        con_map = inverse_data.con_map
        # Flip sign of opt val if maximize.
        opt_val = solution.opt_val
        if solution.status not in s.ERROR and not inverse_data.minimize:
            opt_val = -solution.opt_val

        if solution.status in s.SOLUTION_PRESENT:
            primal_vars = {}
            dual_vars = {}
            # Split vectorized variable into components.
            x = solution.primal_vars.values()[0]
            for var_id, offset in var_map.items():
                var_shape = inverse_data.var_shapes[var_id]
                var_size = np.prod(var_shape)
                primal_vars[var_id] = np.reshape(x[offset:offset+var_size], var_shape,
                                                 order='F')
            # Remap dual variables.
            for old_con, new_con in con_map.items():
                dual_vars[old_con] = solution.dual_vars[new_con]
        else:
            primal_vars = None
            dual_vars = None
        return Solution(solution.status, opt_val,
                        primal_vars, dual_vars)
