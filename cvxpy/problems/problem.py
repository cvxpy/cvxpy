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
from .. import utilities as u
from .. import interface as intf
from ..utilities.ordered_set import OrderedSet
from ..expressions.expression import Expression
from ..expressions.constants import Constant
from ..expressions.variables import Variable
from ..constraints import EqConstraint, LeqConstraint, \
                          SOC, SDP, NonlinearConstraint
from .objective import Minimize, Maximize
from kktsolver import get_kktsolver

from collections import OrderedDict
import itertools
import numbers
import cvxopt
import cvxopt.solvers
import ecos
import scs
# ECHU: ECOS now depends on numpy
import numpy as np
import scipy.sparse as sp

class Problem(u.Canonical):
    """A convex optimization problem.

    Attributes:
        objective: The expression to minimize or maximize.
        constraints: The constraints on the problem variables.
    """

    # The solve methods available.
    REGISTERED_SOLVE_METHODS = {}
    # Interfaces for interacting with matrices.
    _SPARSE_INTF = intf.DEFAULT_SPARSE_INTERFACE
    _DENSE_INTF = intf.DEFAULT_INTERFACE

    def __init__(self, objective, constraints=None):
        if constraints is None:
            constraints = []
        self.objective = objective
        self.constraints = constraints

    def is_dcp(self):
        """Does the problem satisfy DCP rules?
        """
        return all(exp.is_dcp() for exp in self.constraints + [self.objective])

    def _filter_constraints(self, constraints):
        """Separate the constraints by type.

        Args:
            constraints: A list of constraints.

        Returns:
            A map of type key to an ordered set of constraints.
        """
        constr_map = {s.EQ: OrderedSet([]),
                      s.INEQ: OrderedSet([]),
                      s.SOC: OrderedSet([]),
                      s.SDP: OrderedSet([]),
                      s.NONLIN: OrderedSet([])}
        for c in constraints:
            if isinstance(c, EqConstraint):
                constr_map[s.EQ].add(c)
            elif isinstance(c, LeqConstraint):
                constr_map[s.INEQ].add(c)
            elif isinstance(c, SOC):
                constr_map[s.SOC].add(c)
            elif isinstance(c, SDP):
                constr_map[s.SDP].add(c)
            elif isinstance(c, NonlinearConstraint):
                constr_map[s.NONLIN].add(c)
        return constr_map

    def canonicalize(self):
        """Computes the graph implementation of the problem.

        Returns:
            A tuple (affine objective, constraints list)
            and the cone dimensions.
        """
        constraints = []
        obj, constr = self.objective.canonical_form
        constraints += constr
        unique_constraints = list(set(self.constraints))
        for constr in unique_constraints:
            constraints += constr.canonical_form[1]
        constr_map = self._filter_constraints(constraints)
        dims = {'l': sum(c.size[0]*c.size[1] for c in constr_map[s.INEQ])}
        # Formats SOC and SDP constraints for the solver.
        for constr in itertools.chain(constr_map[s.SOC], constr_map[s.SDP]):
            for ineq_constr in constr.format():
                constr_map[s.INEQ].add(ineq_constr)
        dims['q'] = [c.size[0] for c in constr_map[s.SOC]]
        dims['s'] = [c.size[0] for c in constr_map[s.SDP]]
        return (obj, constr_map, dims)

    def variables(self):
        """Returns a list of the variables in the problem.
        """
        vars_ = self.objective.variables()
        for constr in self.constraints:
            vars_ += constr.variables()
        # Remove duplicates.
        return list(set(vars_))

    def parameters(self):
        """Returns a list of the parameters in the problem.
        """
        params = self.objective.parameters()
        for constr in self.constraints:
            params += constr.parameters()
        # Remove duplicates.
        return list(set(params))

    def solve(self, *args, **kwargs):
        """Solves the problem using the specified method.

        Args:
            method: The solve method to use.
            solver: The solver to use. Defaults to ECOS.
            verbose: Overrides the default of hiding solver output.

        Returns:
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        func_name = kwargs.pop("method", None)
        if func_name is not None:
            func = Problem.REGISTERED_SOLVE_METHODS[func_name]
            return func(self, *args, **kwargs)
        else:
            return self._solve(*args, **kwargs)

    @classmethod
    def register_solve(cls, name, func):
        """Adds a solve method to the Problem class.

        Args:
            name: The keyword for the method.
            func: The function that executes the solve method.
        """
        cls.REGISTERED_SOLVE_METHODS[name] = func

    def _solve(self, solver=s.ECOS, ignore_dcp=False, verbose=False):
        """Solves a DCP compliant optimization problem.

        Saves the values of primal and dual variables in the variable
        and constraint objects, respectively.

        Args:
            solver: The solver to use. Defaults to ECOS.
            ignore_dcp: Overrides the default of raising an exception if
                        the problem is not DCP.
            verbose: Overrides the default of hiding solver output.

        Returns:
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        if not self.is_dcp():
            if ignore_dcp:
                print ("Problem does not follow DCP rules. "
                       "Solving a convex relaxation.")
            else:
                raise Exception("Problem does not follow DCP rules.")
        objective, constr_map, dims = self.canonicalize()

        all_ineq = itertools.chain(constr_map[s.EQ], constr_map[s.INEQ])
        var_offsets, x_length = self._get_var_offsets(objective, all_ineq)

        c, obj_offset = self._constr_matrix([objective], var_offsets, x_length,
                                            self._DENSE_INTF,
                                            self._DENSE_INTF)
        A, b = self._constr_matrix(constr_map[s.EQ], var_offsets, x_length,
                                   self._SPARSE_INTF, self._DENSE_INTF)
        G, h = self._constr_matrix(constr_map[s.INEQ], var_offsets, x_length,
                                   self._SPARSE_INTF, self._DENSE_INTF)

        # Save original cvxopt solver options.
        old_options = cvxopt.solvers.options
        # Silence cvxopt if verbose is False.
        cvxopt.solvers.options['show_progress'] = verbose
        # Always do one step of iterative refinement after solving KKT system.
        cvxopt.solvers.options['refinement'] = 1
        # Target cvxopt clp if nonlinear constraints exist
        if constr_map[s.NONLIN]:
            # Get the nonlinear constraints.
            F = self._merge_nonlin(constr_map[s.NONLIN], var_offsets, x_length)
            # Get custom kktsolver.
            kktsolver = get_kktsolver(G, dims, A, F)
            results = cvxopt.solvers.cpl(c.T, F, G, h, A=A, b=b,
                                         dims=dims,kktsolver=kktsolver)
            status = s.SOLVER_STATUS[s.CVXOPT][results['status']]
            primal_val = results['primal objective']
        # Target cvxopt solver if SDP or invalid for ECOS.
        elif solver == s.CVXOPT or len(dims['s']) > 0 or min(G.size) == 0:
            # Get custom kktsolver.
            kktsolver = get_kktsolver(G, dims, A)
            # Adjust tolerance to account for regularization.
            cvxopt.solvers.options['feastol'] = 2*1e-6
            results = cvxopt.solvers.conelp(c.T, G, h, A=A, b=b,
                                            dims=dims, kktsolver=kktsolver)
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
            cnp, hnp, bnp = (np.fromiter(iter(x),
                                        dtype=np.double,
                                        count=len(x))
                             for x in (c, h, b))
            Gp, Gi, Gx = G.CCS
            m, n1 = G.size
            Ap, Ai, Ax = A.CCS
            p, n2 = A.size
            Gp, Gi, Ap, Ai = (np.fromiter(iter(x),
                                         dtype=np.int32,
                                         count=len(x))
                              for x in (Gp, Gi, Ap, Ai))
            Gx, Ax = (np.fromiter(iter(x),
                                  dtype=np.double,
                                  count=len(x))
                      for x in (Gx, Ax))
            Gsp = sp.csc_matrix((Gx, Gi, Gp), shape=(m, n1))

            data = {"c": cnp}
            if p == 0:
                Asp = None
                bnp = None
                # SCS
                dims["f"] = 0
                data["A"] = Gsp
                data["b"] = hnp
            else:
                Asp = sp.csc_matrix((Ax, Ai, Ap), shape=(p, n2))
                # SCS
                dims["f"] = Asp.shape[0]
                data["A"] = sp.vstack([Asp, Gsp])
                data["b"] = np.vstack([bnp, hnp])

            # ECHU: end conversion
            # results = ecos.solve(cnp, Gsp, hnp, dims, Asp, bnp, verbose=verbose)
            # status = s.SOLVER_STATUS[s.ECOS][results['info']['exitFlag']]
            # primal_val = results['info']['pcost']

            results = scs.solve(data, dims)
            status = s.SOLVER_STATUS[s.SCS][results['info']['status']]
            primal_val = results['info']['pobj']
            if status == s.SOLVED:
                self._save_values(results['x'], var_offsets.keys())
                all_ineq = itertools.chain(constr_map[s.EQ], constr_map[s.INEQ])
                self._save_values(results['y'], all_ineq)
                return self.objective._primal_to_result(primal_val - obj_offset[0])

        # Restore original cvxopt solver options.
        cvxopt.solvers.options = old_options

        if status == s.SOLVED:
            self._save_values(results['x'], var_offsets.keys())
            self._save_values(results['y'], constr_map[s.EQ])
            if constr_map[s.NONLIN]:
                self._save_values(results['zl'], constr_map[s.INEQ])
            else:
                self._save_values(results['z'], constr_map[s.INEQ])
            return self.objective._primal_to_result(primal_val - obj_offset[0])
        else:
            return status

    def _get_var_offsets(self, objective, constraints):
        """Maps each variable to a horizontal offset.

        Args:
            objective: The canonicalized objective.
            constraints: The canonicalized constraints.

        Returns:
            A tuple (map of variable to offset, length of variable vector).
        """
        vars_ = objective.variables()
        for constr in constraints:
            vars_ += constr.variables()
        var_offsets = OrderedDict()
        vert_offset = 0
        for var in set(vars_):
            var_offsets[var] = vert_offset
            vert_offset += var.size[0]*var.size[1]
        return (var_offsets, vert_offset)

    def _save_values(self, result_vec, objects):
        """Saves the values of the optimal primal/dual variables.

        Args:
            results_vec: A vector containing the variable values.
            objects: The variables or constraints where the values
                     will be stored.
        """
        if len(result_vec) > 0:
            # Cast to desired matrix type.
            result_vec = self._DENSE_INTF.const_to_matrix(result_vec)
        offset = 0
        for obj in objects:
            rows,cols = obj.size
            # Handle scalars
            if (rows,cols) == (1,1):
                value = intf.index(result_vec, (offset, 0))
            else:
                value = self._DENSE_INTF.zeros(rows, cols)
                self._DENSE_INTF.block_add(value,
                    result_vec[offset:offset + rows*cols],
                    0, 0, rows, cols)
            obj.save_value(value)
            offset += rows*cols

    def _constr_matrix(self, aff_expressions, var_offsets, x_length,
                       matrix_intf, vec_intf):
        """Returns a matrix and vector representing a list of constraints.

        In the matrix, each constraint is given a block of rows.
        Each variable coefficient is inserted as a block with upper
        left corner at matrix[variable offset, constraint offset].
        The constant term in the constraint is added to the vector.

        Args:
            aff_expressions: A list of affine expressions or constraints.
            var_offsets: A dict of variable id to horizontal offset.
            x_length: The length of the x vector.
            matrix_intf: The matrix interface to use for creating the constraints matrix.
            vec_intf: The matrix interface to use for creating the constant vector.

        Returns:
            A (matrix, vector) tuple.
        """

        rows = sum([aff.size[0] * aff.size[1] for aff in aff_expressions])
        cols = x_length
        V, I, J = [], [], []
        const_vec = vec_intf.zeros(rows, 1)
        vert_offset = 0
        for aff_exp in aff_expressions:
            coefficients = aff_exp.coefficients()
            for var, blocks in coefficients.items():
                # Constant is not in var_offsets.
                horiz_offset = var_offsets.get(var)
                for col, block in enumerate(blocks):
                    vert_start = vert_offset + col*aff_exp.size[0]
                    vert_end = vert_start + aff_exp.size[0]
                    if var is s.CONSTANT:
                        const_vec[vert_start:vert_end, :] = block
                    else:
                        if isinstance(block, numbers.Number):
                            V.append(block)
                            I.append(vert_start)
                            J.append(horiz_offset)
                        else: # Block is a matrix or spmatrix.
                            if isinstance(block, cvxopt.matrix):
                                block = cvxopt.sparse(block)
                            V.extend(block.V)
                            I.extend(block.I + vert_start)
                            J.extend(block.J + horiz_offset)
            vert_offset += aff_exp.size[0]*aff_exp.size[1]

        # Create the constraints matrix.
        if len(V) > 0:
            matrix = cvxopt.spmatrix(V, I, J, (rows, cols), tc='d')
            # Convert the constraints matrix to the correct type.
            matrix = matrix_intf.const_to_matrix(matrix, convert_scalars=True)
        else: # Empty matrix.
            matrix = matrix_intf.zeros(rows, cols)
        return (matrix, -const_vec)

    def _merge_nonlin(self, nl_constr, var_offsets, x_length):
        """ TODO: ensure that this works with numpy data structs...
        """
        rows = sum([constr.size[0] * constr.size[1] for constr in nl_constr])
        cols = x_length

        big_x = self._DENSE_INTF.zeros(cols, 1)
        for constr in nl_constr:
            constr.place_x0(big_x, var_offsets, self._DENSE_INTF)

        def F(x=None, z=None):
            if x is None:
                return rows, big_x
            big_f = self._DENSE_INTF.zeros(rows, 1)
            big_Df = self._SPARSE_INTF.zeros(rows, cols)
            if z:
                big_H = self._SPARSE_INTF.zeros(cols, cols)

            offset = 0
            for constr in nl_constr:
                local_x = constr.extract_variables(x, var_offsets,
                                                   self._DENSE_INTF)
                if z:
                    f, Df, H = constr.f(local_x,
                                        z[offset:offset + constr.size[0]])
                else:
                    result = constr.f(local_x)
                    if result:
                        f, Df = result
                    else:
                        return None
                big_f[offset:offset + constr.size[0]] = f
                constr.place_Df(big_Df, Df, var_offsets,
                                offset, self._SPARSE_INTF)
                if z:
                    constr.place_H(big_H, H, var_offsets, self._SPARSE_INTF)
                offset += constr.size[0]

            if z is None:
                return big_f, big_Df
            return big_f, big_Df, big_H
        return F

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Problem(%s, %s)" % (repr(self.objective),
                                    repr(self.constraints))
