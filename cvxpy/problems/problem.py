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

from .. import settings as s
from .. import interface as intf
from ..expressions.expression import Expression
from ..expressions.variables import Variable
from ..constraints.affine import AffEqConstraint, AffLeqConstraint
from ..constraints.second_order import SOC
from ..constraints.nonlinear import NonlinearConstraint

import cvxopt
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
        constr_map = {}
        constr_map[s.EQ] = [c for c in constraints if isinstance(c, AffEqConstraint)]
        constr_map[s.INEQ] = [c for c in constraints if isinstance(c, AffLeqConstraint)]
        constr_map[s.SOC] = [c for c in constraints if isinstance(c, SOC)]
        constr_map[s.NONLIN] = [c for c in constraints if isinstance(c, NonlinearConstraint)]
        return constr_map

    # Convert the problem into an affine objective and affine constraints.
    # Also returns the dimensions of the cones for the solver.
    def canonicalize(self):
        obj,constraints = self.objective.canonical_form()
        for constr in self.constraints:
            constraints += constr.canonical_form()[1]
        constr_map = self.filter_constraints(constraints)
        dims = {'l': sum(c.size[0]*c.size[1] for c in constr_map[s.INEQ])}
        # Formats SOC constraints for the solver.
        for constr in constr_map[s.SOC]:
            constr_map[s.INEQ] += constr.format()
        dims['q'] = [c.size for c in constr_map[s.SOC]]
        dims['s'] = []
        return (obj,constr_map,dims)

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
        objective,constr_map,dims = self.canonicalize()
        variables = self.variables(objective, constr_map[s.EQ] + constr_map[s.INEQ])
        var_ids = self.variable_ids(variables)
       
        c = self.constraints_matrix([objective], var_ids, 
                                    self.dense_interface).T
        A = self.constraints_matrix(constr_map[s.EQ], var_ids, self.interface)
        b = -self.constraints_matrix(constr_map[s.EQ], self.CONSTANT_ID, 
                                     self.dense_interface)
        G = self.constraints_matrix(constr_map[s.INEQ], var_ids, self.interface)
        h = -self.constraints_matrix(constr_map[s.INEQ], self.CONSTANT_ID, 
                                     self.dense_interface)

        # ECHU: get the nonlinear constraints
        F = self.nonlinear_constraint_function(constr_map[s.NONLIN], var_ids, self.interface)

        if constr_map[s.NONLIN]:
            # Target cvxopt clp if nonlinear constraints exist
            results = cvxopt.solvers.cpl(c,F,G,h,A=A,b=b,dims=dims)
            status = s.SOLVER_STATUS[s.CVXOPT][results['status']]
            primal_val = results['primal objective']
        elif solver == s.CVXOPT or len(dims['s']) > 0 or min(G.size) == 0:
            # Target cvxopt solver if SDP or invalid for ECOS.
            results = cvxopt.solvers.conelp(c,G,h,A=A,b=b,dims=dims)
            status = s.SOLVER_STATUS[s.CVXOPT][results['status']]
            primal_val = results['primal objective']
        else: # If possible, target ECOS.
            # ECHU: ecos interface has changed and no longer relies on CVXOPT
            # as a result, we have to convert cvxopt data structures into
            # numpy arrays
            #
            # ideally, CVXPY would no longer user CVXOPT, except when calling
            # conelp
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
            #results = ecos.ecos(c,G,h,dims,A,b)
            status = s.SOLVER_STATUS[s.ECOS][results['info']['exitFlag']]
            primal_val = results['info']['pcost']
        if status == s.SOLVED:
            self.save_values(results['x'], variables)
            self.save_values(results['y'], constr_map[s.EQ])
            if constr_map[s.NONLIN]:
                self.save_values(results['zl'], constr_map[s.INEQ])
            else:
                self.save_values(results['z'], constr_map[s.INEQ])
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

    def nonlinear_constraint_function(self, nl_funcs, var_ids, interface):
        """ TODO: ensure that this works with numpy data structs...
        """
        rows = sum([func.size[0] * func.size[1] for func in nl_funcs])
        cols = len(var_ids) # All variables are scalar.
        
        vert_offset, big_x = 0, cvxopt.matrix(0., (cols,1))
        for func in nl_funcs:
            # get height?
            m, x0 = func.f()
            func.start = vert_offset    # HACK: set the start location
            vert_offset += m
            indices = []
            for variable in func.vars_involved:
                id = variable.index_id(0,0)
                var_size = variable.size[0]*variable.size[1]
                indices += range(var_ids[id],var_ids[id]+var_size)
            big_x[indices] = x0
            func.indices = indices  # HACK: set the indices of the function
        
        
        def F(x=None, z=None):
            if x is None: return rows, big_x
            big_f, big_Df = cvxopt.matrix(0., (rows,1)), cvxopt.spmatrix(0,[0],[0], size=(rows,cols))
            if z: big_H = cvxopt.spmatrix(0,[0],[0], size=(cols,cols))
            
            for func in nl_funcs:
                if z:
                    f, Df, H = func.f(x[func.indices],z[func.start:func.start + func.size[0]])
                else:
                    result = func.f(x[func.indices])
                    if result:
                        f, Df = result
                    else:
                        return None
                big_f[func.start:func.start + func.size[0]] = f
                big_Df[func.start:func.start + func.size[0], func.indices] = Df
                if z: 
                    big_H[func.indices, func.indices] = H
            
            if z is None: return big_f, big_Df
            return big_f, big_Df, big_H
        return F