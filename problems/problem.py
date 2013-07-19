import cvxpy.settings as s
import cvxpy.interface.matrices as intf
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.constraint import EqConstraint, LeqConstraint
from cvxpy.constraints.soc import SOC

import cvxopt.solvers

class Problem(object):
    """
    An optimization problem.
    """
    # Dummy variable list with the constant key for constructing b and h.
    CONST_VAR = (s.CONSTANT,Variable())

    # objective - the problem objective.
    # constraints - the problem constraints.
    def __init__(self, objective, constraints=[]):
        self.objective = objective
        self.constraints = constraints

    # Does the problem satisfy DCP rules?
    def is_dcp(self):
        return all(exp.is_dcp() for exp in self.constraints + [self.objective])

    # Divide the constraints into separate types.
    def filter_constraints(self, constraints):
        eq_constraints = [c for c in constraints if isinstance(c, EqConstraint)]
        ineq_constraints = [c for c in constraints if isinstance(c, LeqConstraint)]
        soc_constraints = [c for c in constraints if isinstance(c, SOC)]
        return (eq_constraints, ineq_constraints, soc_constraints)

    # Convert the problem into an affine objective and affine constraints.
    # Also returns the dimensions of the cones for the solver.
    def canonicalize(self):
        obj,constraints = self.objective.canonicalize()
        for constr in self.constraints:
            constraints += constr.canonicalize()[1]
        eq_constr,ineq_constr,soc_constr = self.filter_constraints(constraints)
        dims = {'l': sum(c.size()[0] for c in ineq_constr)}
        # Formats SOC constraints for the solver.
        for constr in soc_constr:
            ineq_constr += constr.format()
        dims['q'] = [c.size() for c in soc_constr]
        dims['s'] = []
        return (obj,eq_constr,ineq_constr,dims)

    # Solves the problem and returns the value of the objective.
    # Saves the values of variables.
    def solve(self):
        if not self.is_dcp():
            print "Problem does not follow DCP rules."
        objective,eq_constr,ineq_constr,dims = self.canonicalize()
        variables = self.variables(objective, eq_constr + ineq_constr)

        c = Problem.constraints_matrix([objective], variables).T
        A = Problem.constraints_matrix(eq_constr, variables)
        b = -Problem.constraints_matrix(eq_constr, [Problem.CONST_VAR])
        G = Problem.constraints_matrix(ineq_constr, variables)
        h = -Problem.constraints_matrix(ineq_constr, [Problem.CONST_VAR])

        results = cvxopt.solvers.conelp(c,G,h,A=A,b=b,dims=dims)
        if results['status'] == 'optimal':
            Problem.save_values(results['x'], variables)
            return self.objective.value(results)
        else:
            return results['status']

    # A list of variable name and object, sorted alphabetically.
    def variables(self, objective, constraints):
        vars = objective.variables()
        for constr in constraints:
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
            for i in range(var.rows):
                var.value.append(result_vec[offset+i])
            # Handle scalars
            var.value = var.value if len(var.value) > 1 else var.value[0]
            offset += var.rows


    # Returns a matrix where each variable coefficient is inserted as a block
    # with upper left corner at matrix[variable offset, constraint offset].
    @staticmethod
    def constraints_matrix(aff_expressions, variables):
        rows = sum([aff.size()[0] for aff in aff_expressions])
        cols = sum([obj.size()[0] for (name,obj) in variables])
        matrix = intf.zeros(rows,cols)
        horiz_offset = 0
        for (name, obj) in variables:
            vert_offset = 0
            for aff_exp in aff_expressions:
                coefficients = aff_exp.coefficients()
                if name in coefficients:
                    intf.block_copy(matrix, 
                                    coefficients[name],
                                    horiz_offset, 
                                    vert_offset)
                vert_offset += aff_exp.size()[0]
            horiz_offset += obj.size()[0]
        return matrix