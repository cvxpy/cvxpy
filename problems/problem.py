import cvxpy.settings as s
import cvxpy.interface.matrix_utilities as intf
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
from cvxpy.constraints.second_order import SOC

import cvxopt
import cvxopt.solvers

class Problem(object):
    """
    An optimization problem.
    """
    # objective - the problem objective.
    # constraints - the problem constraints.
    # target_matrix - the matrix type used internally.
    def __init__(self, objective, constraints=[], target_matrix=intf.SPARSE_TARGET):
        self.objective = objective
        self.constraints = constraints
        self.interface = intf.get_matrix_interface(target_matrix)
        self.dense_interface = intf.get_matrix_interface(intf.DENSE_TARGET)

    # Does the problem satisfy DCP rules?
    def is_dcp(self):
        return all(exp.is_dcp() for exp in self.constraints + [self.objective])

    # Divide the constraints into separate types.
    def filter_constraints(self, constraints):
        eq_constraints = [c for c in constraints if isinstance(c, AffEqConstraint)]
        ineq_constraints = [c for c in constraints if isinstance(c, AffLeqConstraint)]
        soc_constraints = [c for c in constraints if isinstance(c, SOC)]
        return (eq_constraints, ineq_constraints, soc_constraints)

    # Convert the problem into an affine objective and affine constraints.
    # Also returns the dimensions of the cones for the solver.
    def canonicalize(self):
        obj,constraints = self.objective.canonicalize()
        for constr in self.constraints:
            constraints += constr.canonicalize()[1]
        eq_constr,ineq_constr,soc_constr = self.filter_constraints(constraints)
        dims = {'l': sum(c.size[0]*c.size[1] for c in ineq_constr)}
        # Formats SOC constraints for the solver.
        for constr in soc_constr:
            ineq_constr += constr.format()
        dims['q'] = [c.size for c in soc_constr]
        dims['s'] = []
        return (obj,eq_constr,ineq_constr,dims)

    # Solves the problem and returns the value of the objective.
    # Saves the values of variables.
    def solve(self):
        if not self.is_dcp():
            print "Problem does not follow DCP rules."
        objective,eq_constr,ineq_constr,dims = self.canonicalize()
        variables = self.variables(objective, eq_constr + ineq_constr)
        var_ids = self.variable_ids(variables)

        c = self.constraints_matrix([objective], var_ids, self.dense_interface).T
        A = self.constraints_matrix(eq_constr, var_ids, self.interface)
        b = -self.constraints_matrix(eq_constr, [s.CONSTANT], self.dense_interface)
        G = self.constraints_matrix(ineq_constr, var_ids, self.interface)
        h = -self.constraints_matrix(ineq_constr, [s.CONSTANT], self.dense_interface)

        results = cvxopt.solvers.conelp(c,G,h,A=A,b=b,dims=dims)
        if results['status'] == 'optimal':
            self.save_values(results['x'], variables)
            self.save_values(results['y'], eq_constr)
            return self.objective.value(results)
        else:
            return results['status']

    # A list of variable objects, sorted alphabetically by id.
    def variables(self, objective, constraints):
        vars = objective.variables()
        for constr in constraints:
            vars = dict(vars.items() + constr.variables().items())
        names = vars.keys()
        names.sort()
        return [vars[k] for k in names]

    # A list of variable ids.
    # Matrix variables are represented as a list of scalar variable views.
    def variable_ids(self, variables):
        var_ids = []
        for var in variables:
            # Column major order.
            for col in range(var.size[1]):
                for row in range(var.size[0]):
                    var_ids.append(var.index_id(row,col))
        return var_ids

    # Saves the values of the optimal primary/dual variables 
    # as fields in the variable/constraint objects.
    def save_values(self, result_vec, objects):
        offset = 0
        for obj in objects:
            rows,cols = obj.size
            # Handle scalars
            if (rows,cols) == (1,1):
                value = result_vec[offset]
            else:
                value = obj.interface.zeros(rows, cols)
                obj.interface.block_copy(value, 
                                         result_vec[offset:offset + rows*cols],
                                         0, 0, rows, cols)
            obj.save_value(value)
            offset += rows*cols

    # Returns a matrix where each variable coefficient is inserted as a block
    # with upper left corner at matrix[variable offset, constraint offset].
    # aff_expressions - a list of affine expressions or constraints.
    # var_ids - a list of variable ids.
    # interface - the matrix interface to use for creating the constraints matrix.
    def constraints_matrix(self, aff_expressions, var_ids, interface):
        rows = sum([aff.size[0] * aff.size[1] for aff in aff_expressions])
        cols = len(var_ids) # All variables are scalar.
        matrix = interface.zeros(rows, cols)
        vert_offset = 0
        for aff_exp in aff_expressions:
            num_entries = aff_exp.size[0] * aff_exp.size[1]
            coefficients = aff_exp.coefficients(interface)
            horiz_offset = 0
            for id in var_ids:
                if id in coefficients:
                    # Update the matrix.
                    interface.block_copy(matrix,
                                         coefficients[id],
                                         vert_offset,
                                         horiz_offset,
                                         num_entries,
                                         1)
                horiz_offset += 1
            vert_offset += num_entries
        return matrix