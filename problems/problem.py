import settings
import cvxopt
import cvxopt.solvers
from expressions.expression import Parameter

class Problem(object):
    """
    An optimization problem.
    """
    # Dummy variable list with the constant key for constructing b and h.
    CONST_VAR = (settings.CONSTANT,Parameter(1))

    # objective - the problem objective.
    # constraints - the problem constraints.
    def __init__(self, objective, constraints=[]):
        self.objective = objective
        self.constraints = constraints

    # Solves the problem and returns the value of the objective.
    # Saves the values of variables.
    def solve(self):
        variables = self.variables()

        c = Problem.constraints_matrix(c, variables)
        eq_constraints = [c for c in self.constraints if c.type == settings.EQ_CONSTR]
        eq_constr_matrices = Problem.linear_op_matrices(eq_constraints, variables)
        ineq_constraints = [c for c in self.constraints if c.type == settings.INEQ_CONSTR]
        ineq_constr_matrices = Problem.linear_op_matrices(ineq_constraints, variables)
        
        A = Problem.constraints_matrix(eq_constraints, variables)
        b = -Problem.constraints_matrix(eq_constraints, [Problem.CONST_VAR])
        G = Problem.constraints_matrix(ineq_constraints, variables)
        h = -Problem.constraints_matrix(ineq_constraints, [Problem.CONST_VAR])
        results = cvxopt.solvers.conelp(c,G,h,A=A,b=b)
        if results['x'] is not None:
            Problem.save_values(results['x'], variables)
            return results['primal objective']

    # A dict of variable name to (objects,offset) for the variables in the problem.
    def variables(self):
        vars = self.objective.variables()
        for constr in self.constraints:
            vars = dict(vars.items() + constr.variables().items())
        names = vars.keys()
        names.sort()
        return [(k,vars[k]) for k in names]

    # Saves the values of the optimal variables 
    # as fields in the variable objects.
    @staticmethod
    def save_values(result_vec, variables):
        for (name,(obj,offset)) in variables.items():
            var.value = []
            for i in range(obj.rows):
                var.value.append(result_vec[var[offset]+i])
            # Handle scalars
            var.value = var.value if len(var.value) > 1 else var.value[0]


    # Returns a matrix where each variable coefficient is inserted as a block
    # with upper left corner at matrix[variable offset, constraint offset].
    @staticmethod
    def constraints_matrix(aff_expressions, variables):
        rows = sum([aff.shape().rows for aff in aff_expressions])
        cols = sum([obj.rows for (name,(obj,offset)) in variables.items()])
        matrix = cvxopt.matrix(0, (rows,cols), 'd') # Real matrix of zeros