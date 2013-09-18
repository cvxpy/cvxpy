"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy.settings as s
import cvxpy.interface.matrix_utilities as intf
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
from cvxpy.constraints.second_order import SOC

import cvxopt.solvers
import ecos
# ECHU: ECOS now depends on numpy
import numpy as np
import scipy.sparse as sp

class Problem(object):
    """
    An optimization problem.
    """
    # The solve methods available.
    REGISTERED_SOLVE_METHODS = {}
    # Simulated variable id dict for constant.
    CONSTANT_ID = {s.CONSTANT:0}
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
    # Remove duplicate constraint objects.
    def filter_constraints(self, constraints):
        constraints = list(set(constraints)) # TODO generalize
        eq_constraints = [c for c in constraints if isinstance(c, AffEqConstraint)]
        ineq_constraints = [c for c in constraints if isinstance(c, AffLeqConstraint)]
        soc_constraints = [c for c in constraints if isinstance(c, SOC)]
        return (eq_constraints, ineq_constraints, soc_constraints)

    # Convert the problem into an affine objective and affine constraints.
    # Also returns the dimensions of the cones for the solver.
    def canonicalize(self):
        obj,constraints = self.objective.canonical_form()
        for constr in self.constraints:
            constraints += constr.canonical_form()[1]
        eq_constr,ineq_constr,soc_constr = self.filter_constraints(constraints)
        dims = {'l': sum(c.size[0]*c.size[1] for c in ineq_constr)}
        # Formats SOC constraints for the solver.
        for constr in soc_constr:
            ineq_constr += constr.format()
        dims['q'] = [c.size for c in soc_constr]
        dims['s'] = []
        return (obj,eq_constr,ineq_constr,dims)

    # Dispatcher for different solve methods.
    def solve(self, *args, **kwargs):
        func_name = kwargs.pop("method", None)
        if func_name is not None:
            func = Problem.REGISTERED_SOLVE_METHODS[func_name]
            return func(self, *args, **kwargs)
        else:
            return self._solve(*args, **kwargs)

    # Register a solve method.
    @staticmethod
    def register_solve(name, func):
        Problem.REGISTERED_SOLVE_METHODS[name] = func

    # Solves DCP compliant optimization problems.
    # Saves the values of primal and dual variables.
    def _solve(self, solver=s.ECOS, ignore_dcp=False):
        if not self.is_dcp():
            if ignore_dcp:
                print ("Problem does not follow DCP rules. "
                       "Solving a convex relaxation.")
            else:
                raise Exception("Problem does not follow DCP rules.")
        objective,eq_constr,ineq_constr,dims = self.canonicalize()
        variables = self.variables(objective, eq_constr + ineq_constr)
        var_ids = self.variable_ids(variables)
       
        c = self.constraints_matrix([objective], var_ids, 
                                    self.dense_interface).T
        A = self.constraints_matrix(eq_constr, var_ids, self.interface)
        b = -self.constraints_matrix(eq_constr, self.CONSTANT_ID, 
                                     self.dense_interface)
        G = self.constraints_matrix(ineq_constr, var_ids, self.interface)
        h = -self.constraints_matrix(ineq_constr, self.CONSTANT_ID, 
                                     self.dense_interface)

        # Target cvxopt solver if SDP or invalid for ECOS.
        if solver == s.CVXOPT or len(dims['s']) > 0 or min(G.size) == 0:
            results = cvxopt.solvers.conelp(c,G,h,A=A,b=b,dims=dims)
            status = s.SOLVER_STATUS[s.CVXOPT][results['status']]
            primal_val = results['primal objective']
        else: # If possible, target ECOS.
            if hasattr(ecos, 'solve'):
                # ECHU: ecos interface has changed and no longer relies on 
                # CVXOPT; as a result, we have to convert cvxopt data 
                # structures into numpy arrays
                #
                # ideally, CVXPY would no longer user CVXOPT, except when 
                # calling conelp
                #
                cnp, hnp, bnp = map(lambda x: np.fromiter(iter(x),dtype=np.double,count=len(x)), (c, h, b))
                Gp,Gi,Gx = G.CCS
                m,n1 = G.size
                Ap,Ai,Ax = A.CCS
                p,n2 = A.size
                Gp, Gi, Ap, Ai = map(lambda x: np.fromiter(iter(x),dtype=np.int32,count=len(x)), (Gp,Gi,Ap,Ai))
                Gx, Ax = map(lambda x: np.fromiter(iter(x),dtype=np.double,count=len(x)), (Gx, Ax))
                Gsp = sp.csc_matrix((Gx,Gi,Gp),shape=(m,n1))
                Asp = sp.csc_matrix((Ax,Ai,Ap),shape=(p,n2))
                # ECHU: end conversion    
                results = ecos.solve(cnp,Gsp,hnp,dims,Asp,bnp)
            else:
                # ECHU: old call to ecos
                results = ecos.ecos(c,G,h,dims,A,b)
            status = s.SOLVER_STATUS[s.ECOS][results['info']['exitFlag']]
            primal_val = results['info']['pcost']
        if status == s.SOLVED:
            self.save_values(results['x'], variables)
            self.save_values(results['y'], eq_constr)
            self.save_values(results['z'], ineq_constr)
            return self.objective.value(primal_val)
        else:
            return status

    # A list of variable objects, sorted alphabetically by id.
    def variables(self, objective, constraints):
        vars = objective.variables()
        for constr in constraints:
            vars += constr.variables()
        # Eliminate duplicate ids and sort variables.
        var_id = {v.id: v for v in vars}
        keys = sorted(var_id.keys())
        return [var_id[k] for k in keys]

    # A dict of variable id to offset in the overall x vector.
    # Matrix variables are represented as a list of scalar variable views.
    def variable_ids(self, variables):
        var_ids = {}
        index = 0
        for var in variables:
            # Column major order.
            for col in range(var.size[1]):
                for row in range(var.size[0]):
                    id = var.index_id(row,col)
                    var_ids[id] = index
                    index += 1
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
    # var_ids - a dict of variable id to offset.
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
            for id,block in coefficients.items():
                if id in var_ids:
                    # Update the matrix.
                    interface.block_copy(matrix,
                                         block,
                                         vert_offset,
                                         var_ids[id],
                                         num_entries,
                                         1)
            vert_offset += num_entries
        return matrix