import settings
import cvxopt
import cvxopt.solvers
from expressions.expression import Parameter

class Problem(object):
    # Dummy variable list with the constant key for constructing b and h.
    CONST_VAR = (settings.CONSTANT,Parameter(1))
    """
    An optimization problem.
    objective - the problem objective.
    constraints - the problem constraints.
    """
    def __init__(self, objective, constraints=[]):
        self.objective = objective
        self.constraints = constraints

    # Solves the problem and returns the value of the objective.
    # Saves the values of variables.
    def solve(self):
        variables = self.variables()
        eq_constraints = [c for c in self.constraints if c.type == settings.EQ_CONSTR]
        ineq_constraints = [c for c in self.constraints if c.type == settings.INEQ_CONSTR]
        c = Problem.constraints_matrix([self.objective], variables)
        A = Problem.constraints_matrix(eq_constraints, variables)
        b = -Problem.constraints_matrix(eq_constraints, [Problem.CONST_VAR])
        G = Problem.constraints_matrix(ineq_constraints, variables)
        h = -Problem.constraints_matrix(ineq_constraints, [Problem.CONST_VAR])
        results = cvxopt.solvers.conelp(c,G,h,A=A,b=b)
        if results['x'] is not None:
            Problem.save_values(results['x'], variables)
            return results['primal objective']

    # A list of the variable names and objects in the problem,
    # ordered alphabetically.
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
        offset = 0
        for (name,var) in variables:
            var.value = []
            cols = var.shape().rows
            for i in range(cols):
                var.value.append(result_vec[offset+i])
            # Handle scalars
            var.value = var.value if len(var.value) > 1 else var.value[0]
            offset += cols


    # Returns a matrix built from the coefficients for each
    # variable in each affine expression, e.g. A in A*x == 0.
    @staticmethod
    def constraints_matrix(aff_expressions, variables):
        rows = sum([aff.shape().rows for aff in aff_expressions])
        cols = sum([var.shape().rows for (name,var) in variables])
        matrix = cvxopt.matrix(0, (rows,cols), 'd') # Real matrix of zeros
        horiz_offset = 0
        for (name,var) in variables:
            width = var.shape().rows
            vert_offset = 0
            for aff in aff_expressions:
                height = aff.shape().rows
                if name in aff.coefficients().coeff_dict: # TODO getter and setter for coeff_dict?
                    coeff = aff.coefficients().coeff_dict.get(name)
                    Problem.add_coefficient(coeff, matrix, width, height, 
                                        horiz_offset, vert_offset)
                vert_offset += height
            horiz_offset += width
        return matrix

    # Writes the coefficient's values to the appropriate space in the matrix.
    @staticmethod
    def add_coefficient(coeff, matrix, rows, cols, horiz_offset, vert_offset):
        # Scalar
        if rows == 1 and cols == 1:
            matrix[vert_offset, horiz_offset] = coeff
        else: # Matrix
            for i in range(rows):
                for j in range(cols):
                    matrix[vert_offset + i, horiz_offset + j] = coeff[i][j]