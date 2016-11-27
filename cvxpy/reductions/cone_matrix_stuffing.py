
from cvxpy.atoms import reshape
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
from cvxpy.expressions.variables import Variable
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.utilities import QuadCoeffExtractor
import cvxpy.settings as s
import numpy as np

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
            all([arg.is_affine() for c in problem.constraints for arg in c.args]))

    # TODO(mwytock): Refactor inversion data and re-use with QPMatrixStuffing
    # which is identical.
    def get_sym_data(self, objective, constraints, cached_data=None):
        class SymData(object):
            def __init__(self, objective, constraints):
                self.constr_map = {s.EQ: constraints}
                vars_ = objective.variables()
                for c in constraints:
                    vars_ += c.variables()
                vars_ = list(set(vars_))
                self.vars_ = vars_
                self.var_offsets, self.var_shapes, self.x_length = self.get_var_offsets(vars_)

            def get_var_offsets(self, variables):
                var_offsets = {}
                var_shapes = {}
                vert_offset = 0
                for x in variables:
                    var_shapes[x.id] = x.shape
                    var_offsets[x.id] = vert_offset
                    vert_offset += x.shape[0]*x.shape[1]
                return (var_offsets, var_shapes, vert_offset)

        return SymData(objective, constraints)

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.
        """
        objective = problem.objective
        constraints = problem.constraints

        sym_data = self.get_sym_data(objective, constraints)

        id_map = sym_data.var_offsets
        N = sym_data.x_length

        extractor = QuadCoeffExtractor(id_map, N)

        # Extract the coefficients
        _, C, R = extractor.get_coeffs(objective.args[0])
        c = np.asarray(C.todense()).flatten()
        r = R[0]
        x = Variable(N)
        if type(objective) == Minimize:
            new_obj = c.T*x + r
        else:
            new_obj = (-c).T*x + -r
        # Form the constraints
        new_cons = []
        for con in constraints:
            arg_list = []
            for arg in con.args:
                _, A, b = extractor.get_coeffs(arg)
                arg_list.append(reshape(A*x + b, arg.shape))
            new_cons.append(type(con)(*arg_list))

        new_prob = Problem(Minimize(new_obj), new_cons)
        return (new_prob, sym_data)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
# TODO: not finished
# primal_vars: dict of id to numpy ndarray
# dual_vars: dict of id to numpy ndarray
# opt_val:
# status
        id_map = inverse_data.var_offsets
        N = inverse_data.x_length

        x = solution.primal_vars
        lmb = solution.dual_vars[0]
        nu = solution.dual_vars[1]
        status = solution.status
        ret = {
            "primal_vars": None,
            "dual_vars": None,
            "opt_val": None,
            "status": status
        }
        if status == "Optimal":
            ret['opt_val'] = solution['']
            ret.primal_vars = x
            ret.dual_vars = (lmb, nu)
        return ret
