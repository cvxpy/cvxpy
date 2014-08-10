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
import cvxpy.utilities as u
from toolz.itertoolz import unique
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.lin_ops as lo
import cvxpy.lin_ops.lin_to_matrix as op2mat
import cvxpy.lin_ops.tree_mat as tree_mat
from cvxpy.constraints import (EqConstraint, LeqConstraint,
SOC, SOC_Elemwise, SDP, ExpCone)
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.problems.kktsolver import get_kktsolver
import cvxpy.problems.iterative as iterative

from collections import OrderedDict
import warnings
import itertools
import numbers
import cvxopt
import cvxopt.solvers
import ecos
import numpy as np
import scipy.sparse as sp

# Attempt to import SCS.
try:
    import scs
except ImportError:
    warnings.warn("The solver SCS could not be imported.")

class Problem(u.Canonical):
    """A convex optimization problem.

    Attributes
    ----------
    objective : Minimize or Maximize
        The expression to minimize or maximize.
    constraints : list
        The constraints on the problem variables.
    """

    # The solve methods available.
    REGISTERED_SOLVE_METHODS = {}
    # Interfaces for interacting with matrices.
    _SPARSE_INTF = intf.DEFAULT_SPARSE_INTERFACE
    _DENSE_INTF = intf.DEFAULT_INTERFACE
    _CVXOPT_DENSE_INTF = intf.get_matrix_interface(cvxopt.matrix)
    _CVXOPT_SPARSE_INTF = intf.get_matrix_interface(cvxopt.spmatrix)

    def __init__(self, objective, constraints=None):
        if constraints is None:
            constraints = []
        # Check that objective is Minimize or Maximize.
        if not isinstance(objective, (Minimize, Maximize)):
            raise TypeError("Problem objective must be Minimize or Maximize.")
        self.objective = objective
        self.constraints = constraints
        self._value = None
        self._status = None

    @property
    def value(self):
        """The value from the last time the problem was solved.

        Returns
        -------
        float or None
        """
        return self._value

    @property
    def status(self):
        """The status from the last time the problem was solved.

        Returns
        -------
        str
        """
        return self._status

    def is_dcp(self):
        """Does the problem satisfy DCP rules?
        """
        return all(exp.is_dcp() for exp in self.constraints + [self.objective])

    def _filter_constraints(self, constraints):
        """Separate the constraints by type.

        Parameters
        ----------
        constraints : list
            A list of constraints.

        Returns
        -------
        dict
            A map of type key to an ordered set of constraints.
        """
        constr_map = {s.EQ: [],
                      s.LEQ: [],
                      s.SOC: [],
                      s.SOC_EW: [],
                      s.SDP: [],
                      s.EXP: []}
        for c in constraints:
            if isinstance(c, lo.LinEqConstr):
                constr_map[s.EQ].append(c)
            elif isinstance(c, lo.LinLeqConstr):
                constr_map[s.LEQ].append(c)
            elif isinstance(c, SOC):
                constr_map[s.SOC].append(c)
            elif isinstance(c, SDP):
                constr_map[s.SDP].append(c)
            elif isinstance(c, ExpCone):
                constr_map[s.EXP].append(c)
        return constr_map

    def canonicalize(self):
        """Computes the graph implementation of the problem.

        Returns
        -------
        tuple
            (affine objective,
             constraints dict)
        """
        canon_constr = []
        obj, constr = self.objective.canonical_form
        canon_constr += constr

        for constr in self.constraints:
            canon_constr += constr.canonical_form[1]
        # Remove redundant constraints.
        canon_constr = unique(canon_constr,
                              key=lambda c: c.constr_id)
        constr_map = self._filter_constraints(canon_constr)

        return (obj, constr_map)

    def _format_for_solver(self, constr_map, solver):
        """Formats the problem for the solver.

        Parameters
        ----------
        constr_map : dict
            A map of constraint type to a list of constraints.
        solver: str
            The solver being targetted.

        Returns
        -------
        dict
            The dimensions of the cones.
        """
        # Initialize dimensions.
        dims = {}
        dims[s.EQ_DIM] = sum(c.size[0]*c.size[1] for c in constr_map[s.EQ])
        dims[s.LEQ_DIM] = sum(c.size[0]*c.size[1] for c in constr_map[s.LEQ])
        dims[s.SOC_DIM] = []
        dims[s.SDP_DIM] = []
        dims[s.EXP_DIM] = 0
        # Formats SOC, SOC_EW, SDP, and EXP constraints for the solver.
        nonlin = constr_map[s.SOC] + constr_map[s.SDP] + constr_map[s.EXP]
        for constr in nonlin:
            constr.format(constr_map[s.EQ], constr_map[s.LEQ], dims, solver)

        return dims

    @staticmethod
    def _constraints_count(constr_map):
        """Returns the number of internal constraints.
        """
        return sum([len(cset) for cset in constr_map.values()])

    def _choose_solver(self, constr_map):
        """Determines the appropriate solver.

        Parameters
        ----------
        constr_map: dict
            A dict of the canonicalized constraints.

        Returns
        -------
        str
            The solver that will be used.
        """
        # If no constraints, use ECOS.
        if self._constraints_count(constr_map) == 0:
            return s.ECOS
        # If SDP or EXP, defaults to CVXOPT.
        elif constr_map[s.SDP] or constr_map[s.EXP]:
            return s.CVXOPT
        # Otherwise use ECOS.
        else:
            return s.ECOS

    def _validate_solver(self, constr_map, solver):
        """Raises an exception if the solver cannot solve the problem.

        Parameters
        ----------
        constr_map: dict
            A dict of the canonicalized constraints.
        solver : str
            The solver to be used.
        """
        if (constr_map[s.SDP] and not solver in s.SDP_CAPABLE) or \
           (constr_map[s.EXP] and not solver in s.EXP_CAPABLE) or \
           (self._constraints_count(constr_map) == 0 and solver == s.SCS):
            raise Exception(
                "The solver %s cannot solve the problem." % solver
            )

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

        Parameters
        ----------
        method : function
            The solve method to use.
        solver : str, optional
            The solver to use.
        verbose : bool, optional
            Overrides the default of hiding solver output.
        solver_specific_opts : dict, optional
            A dict of options that will be passed to the specific solver.
            In general, these options will override any default settings
            imposed by cvxpy.

        Returns
        -------
        float
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

        Parameters
        ----------
        name : str
            The keyword for the method.
        func : function
            The function that executes the solve method.
        """
        cls.REGISTERED_SOLVE_METHODS[name] = func

    def get_problem_data(self, solver):
        """Returns the problem data used in the call to the solver.

        Parameters
        ----------
        solver : str
            The solver the problem data is for.

        Returns
        -------
        tuple
            arguments to solver
        """
        objective, constr_map = self.canonicalize()
        # Raise an error if the solver cannot handle the problem.
        self._validate_solver(constr_map, solver)
        dims = self._format_for_solver(constr_map, solver)
        all_ineq = constr_map[s.EQ] + constr_map[s.LEQ]
        var_offsets, var_sizes, x_length = self._get_var_offsets(objective,
                                                                 all_ineq)

        if solver == s.ECOS and not (constr_map[s.SDP] or constr_map[s.EXP]):
            args, offset = self._ecos_problem_data(objective, constr_map, dims,
                                                   var_offsets, x_length)
        elif solver == s.CVXOPT and not constr_map[s.EXP]:
            args, offset = self._cvxopt_problem_data(objective, constr_map, dims,
                                                     var_offsets, x_length)
        elif solver == s.SCS:
            args, offset = self._scs_problem_data(objective, constr_map, dims,
                                                  var_offsets, x_length)
        else:
            raise Exception("Cannot return problem data for the solver %s." % solver)
        return args

    def _solve(self, solver=None, ignore_dcp=False, verbose=False,
               solver_specific_opts=None):
        """Solves a DCP compliant optimization problem.

        Saves the values of primal and dual variables in the variable
        and constraint objects, respectively.

        Parameters
        ----------
        solver : str, optional
            The solver to use. Defaults to ECOS.
        ignore_dcp : bool, optional
            Overrides the default of raising an exception if the problem is not
            DCP.
        verbose : bool, optional
            Overrides the default of hiding solver output.
        solver_specific_opts : dict, optional
            A dict of options that will be passed to the specific solver.
            In general, these options will override any default settings
            imposed by cvxpy.

        Returns
        -------
        float
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        # Safely set default as empty dict.
        if solver_specific_opts is None:
            solver_specific_opts = {}

        if not self.is_dcp():
            if ignore_dcp:
                print ("Problem does not follow DCP rules. "
                       "Solving a convex relaxation.")
            else:
                raise Exception("Problem does not follow DCP rules.")

        objective, constr_map = self.canonicalize()
        # Choose a default solver if none specified.
        if solver is None:
            solver = self._choose_solver(constr_map)
        else:
            # Raise an error if the solver cannot handle the problem.
            self._validate_solver(constr_map, solver)

        dims = self._format_for_solver(constr_map, solver)

        all_ineq = constr_map[s.EQ] + constr_map[s.LEQ]
        # CVXOPT can have variables that only live in NonLinearConstraints.
        nonlinear = []
        if solver == s.CVXOPT:
            nonlinear = constr_map[s.EXP]
        var_offsets, var_sizes, x_length = self._get_var_offsets(objective,
                                                                 all_ineq,
                                                                 nonlinear)

        if solver == s.CVXOPT:
            result = self._cvxopt_solve(objective, constr_map, dims,
                                        var_offsets, x_length,
                                        verbose, solver_specific_opts)
        elif solver == s.SCS:
            result = self._scs_solve(objective, constr_map, dims,
                                     var_offsets, x_length,
                                     verbose, solver_specific_opts)
        elif solver == s.ECOS:
            result = self._ecos_solve(objective, constr_map, dims,
                                      var_offsets, x_length,
                                      verbose, solver_specific_opts)
        else:
            raise Exception("Unknown solver.")

        status, value, x, y, z = result
        if status == s.OPTIMAL:
            self._save_values(x, self.variables(), var_offsets)
            self._save_dual_values(y, constr_map[s.EQ], EqConstraint)
            self._save_dual_values(z, constr_map[s.LEQ], LeqConstraint)
            self._value = value
        else:
            self._handle_failure(status)
        self._status = status
        return self.value

    def _ecos_problem_data(self, objective, constr_map, dims,
                           var_offsets, x_length):
        """Returns the problem data for the call to ECOS.

        Parameters
        ----------
            objective: Expression
                The canonicalized objective.
            constr_map: dict
                A dict of the canonicalized constraints.
            dims: dict
                A dict with information about the types of constraints.
            var_offsets: dict
                A dict mapping variable id to offset in the stacked variable x.
            x_length: int
                The height of x.
        Returns
        -------
        tuple
            ((c, G, h, dims, A, b), offset)
        """
        c, obj_offset = self._get_obj(objective, var_offsets, x_length,
                                      self._DENSE_INTF,
                                      self._DENSE_INTF)
        # Convert obj_offset to a scalar.
        obj_offset = self._DENSE_INTF.scalar_value(obj_offset)

        A, b = self._constr_matrix(constr_map[s.EQ], var_offsets, x_length,
                                   self._SPARSE_INTF, self._DENSE_INTF)
        G, h = self._constr_matrix(constr_map[s.LEQ], var_offsets, x_length,
                                   self._SPARSE_INTF, self._DENSE_INTF)
        # Convert c,h,b to 1D arrays.
        c, h, b = map(intf.from_2D_to_1D, [c.T, h, b])
        # Return the arguments that would be passed to ECOS.
        return ((c, G, h, dims, A, b), obj_offset)

    def _ecos_solve(self, objective, constr_map, dims,
                    var_offsets, x_length,
                    verbose, opts):
        """Calls the ECOS solver and returns the result.

        Parameters
        ----------
            objective: Expression
                The canonicalized objective.
            constr_map: dict
                A dict of the canonicalized constraints.
            dims: dict
                A dict with information about the types of constraints.
            var_offsets: dict
                A dict mapping variable id to offset in the stacked variable x.
            x_length: int
                The height of x.
            verbose: bool
                Should the solver show output?
            opts: dict
                List of user-specific options for ECOS
        Returns
        -------
        tuple
            (status, optimal objective, optimal x,
             optimal equality constraint dual,
             optimal inequality constraint dual)

        """
        prob_data = self._ecos_problem_data(objective, constr_map, dims,
                                            var_offsets, x_length)
        obj_offset = prob_data[1]
        results = ecos.solve(*prob_data[0], verbose=verbose)
        status = s.SOLVER_STATUS[s.ECOS][results['info']['exitFlag']]
        if status == s.OPTIMAL:
            primal_val = results['info']['pcost']
            value = self.objective._primal_to_result(
                          primal_val - obj_offset)
            return (status, value,
                    results['x'], results['y'], results['z'])
        else:
            return (status, None, None, None, None)

    def _cvxopt_problem_data(self, objective, constr_map, dims,
                             var_offsets, x_length):
        """Returns the problem data for the call to CVXOPT.

        Assumes no exponential cone constraints.

        Parameters
        ----------
            objective: Expression
                The canonicalized objective.
            constr_map: dict
                A dict of the canonicalized constraints.
            dims: dict
                A dict with information about the types of constraints.
            var_offsets: dict
                A dict mapping variable id to offset in the stacked variable x.
            x_length: int
                The height of x.
        Returns
        -------
        tuple
            ((c, G, h, dims, A, b), offset)
        """
        c, obj_offset = self._get_obj(objective, var_offsets, x_length,
                                      self._CVXOPT_DENSE_INTF,
                                      self._CVXOPT_DENSE_INTF)
        # Convert obj_offset to a scalar.
        obj_offset = self._CVXOPT_DENSE_INTF.scalar_value(obj_offset)

        A, b = self._constr_matrix(constr_map[s.EQ], var_offsets, x_length,
                                   self._CVXOPT_SPARSE_INTF,
                                   self._CVXOPT_DENSE_INTF)

        G, h = self._constr_matrix(constr_map[s.LEQ], var_offsets, x_length,
                                   self._CVXOPT_SPARSE_INTF,
                                   self._CVXOPT_DENSE_INTF)
        # Return the arguments that would be passed to CVXOPT.
        return ((c.T, G, h, dims, A, b), obj_offset)


    def _cvxopt_solve(self, objective, constr_map, dims,
                      var_offsets, x_length,
                      verbose, opts):
        """Calls the CVXOPT conelp or cpl solver and returns the result.

        Parameters
        ----------
            objective: Expression
                The canonicalized objective.
            constr_map: dict
                A dict of the canonicalized constraints.
            dims: dict
                A dict with information about the types of constraints.
            sorted_vars: list
                An ordered list of the problem variables.
            var_offsets: dict
                A dict mapping variable id to offset in the stacked variable x.
            x_length: int
                The height of x.
            verbose: bool
                Should the solver show output?
            opts: dict
                List of user-specific options for CVXOPT;
                will be inserted into cvxopt.solvers.options.

        Returns
        -------
        tuple
            (status, optimal objective, optimal x,
             optimal equality constraint dual,
             optimal inequality constraint dual)

        """
        prob_data = self._cvxopt_problem_data(objective, constr_map, dims,
                                              var_offsets, x_length)
        c, G, h, dims, A, b = prob_data[0]
        obj_offset = prob_data[1]
        # Save original cvxopt solver options.
        old_options = cvxopt.solvers.options
        # Silence cvxopt if verbose is False.
        cvxopt.solvers.options['show_progress'] = verbose
        # Always do one step of iterative refinement after solving KKT system.
        cvxopt.solvers.options['refinement'] = 1

        # Apply any user-specific options
        for key, value in opts.items():
            cvxopt.solvers.options[key] = value

        try:
            # Target cvxopt clp if nonlinear constraints exist
            if constr_map[s.EXP]:
                # Get the nonlinear constraints.
                F = self._merge_nonlin(constr_map[s.EXP], var_offsets,
                                       x_length)
                # Get custom kktsolver.
                kktsolver = get_kktsolver(G, dims, A, F)
                results = cvxopt.solvers.cpl(c, F, G, h, dims, A, b,
                                             kktsolver=kktsolver)
            else:
                # Get custom kktsolver.
                kktsolver = get_kktsolver(G, dims, A)
                results = cvxopt.solvers.conelp(c, G, h, dims, A, b,
                                                kktsolver=kktsolver)
            status = s.SOLVER_STATUS[s.CVXOPT][results['status']]
        # Catch exceptions in CVXOPT and convert them to solver errors.
        except ValueError as e:
            status = s.SOLVER_ERROR

        # Restore original cvxopt solver options.
        cvxopt.solvers.options = old_options
        if status == s.OPTIMAL:
            primal_val = results['primal objective']
            value = self.objective._primal_to_result(
                          primal_val - obj_offset)
            if constr_map[s.EXP]:
                ineq_dual = results['zl']
            else:
                ineq_dual = results['z']
            return (status, value, results['x'], results['y'], ineq_dual)
        else:
            return (status, None, None, None, None)

    def _scs_problem_data(self, objective, constr_map, dims,
                          var_offsets, x_length):
        """Returns the problem data for the call to SCS.

        Parameters
        ----------
            objective: Expression
                The canonicalized objective.
            constr_map: dict
                A dict of the canonicalized constraints.
            dims: dict
                A dict with information about the types of constraints.
            var_offsets: dict
                A dict mapping variable id to offset in the stacked variable x.
            x_length: int
                The height of x.
        Returns
        -------
        tuple
            ((data, dims), offset)
        """
        c, obj_offset = self._get_obj(objective, var_offsets, x_length,
                                      self._DENSE_INTF,
                                      self._DENSE_INTF)
        # Convert obj_offset to a scalar.
        obj_offset = self._DENSE_INTF.scalar_value(obj_offset)

        A, b = self._constr_matrix(constr_map[s.EQ] + constr_map[s.LEQ],
                                   var_offsets, x_length,
                                   self._SPARSE_INTF, self._DENSE_INTF)
        # Convert c, b to 1D arrays.
        c, b = map(intf.from_2D_to_1D, [c.T, b])
        data = {"c": c}
        data["A"] = A
        data["b"] = b
        return ((data, dims), obj_offset)

    def _scs_solve(self, objective, constr_map, dims,
                   var_offsets, x_length,
                   verbose, opts):
        """Calls the SCS solver and returns the result.

        Parameters
        ----------
            objective: LinExpr
                The canonicalized objective.
            constr_map: dict
                A dict of the canonicalized constraints.
            dims: dict
                A dict with information about the types of constraints.
            var_offsets: dict
                A dict mapping variable id to offset in the stacked variable x.
            x_length: int
                The height of x.
            verbose: bool
                Should the solver show output?
            opts: dict
                A dict of the solver parameters passed to scs

        Returns
        -------
        tuple
            (status, optimal objective, optimal x,
             optimal equality constraint dual,
             optimal inequality constraint dual)
        """
        prob_data = self._scs_problem_data(objective, constr_map, dims,
                                           var_offsets, x_length)
        obj_offset = prob_data[1]
        # Set the options to be VERBOSE plus any user-specific options.
        opts = dict({ "VERBOSE": verbose }.items() + opts.items())
        use_indirect = opts["USE_INDIRECT"] if "USE_INDIRECT" in opts else False
        results = scs.solve(*prob_data[0], opts=opts, USE_INDIRECT = use_indirect)
        status = s.SOLVER_STATUS[s.SCS][results["info"]["status"]]
        if status == s.OPTIMAL:
            primal_val = results["info"]["pobj"]
            value = self.objective._primal_to_result(primal_val - obj_offset)
            eq_dual = results["y"][0:dims["f"]]
            ineq_dual = results["y"][dims["f"]:]
            return (status, value, results["x"], eq_dual, ineq_dual)
        else:
            return (status, None, None, None, None)

    def _handle_failure(self, status):
        """Updates value fields based on the cause of solver failure.

        Parameters
        ----------
            status: str
                The status of the solver.
        """
        # Set all primal and dual variable values to None.
        for var_ in self.variables():
            var_.save_value(None)
        for constr in self.constraints:
            constr.save_value(None)
        # Set the problem value.
        if status == s.INFEASIBLE:
            self._value = self.objective._primal_to_result(np.inf)
        elif status == s.UNBOUNDED:
            self._value = self.objective._primal_to_result(-np.inf)
        else: # Solver error
            self._value = None

    def _get_var_offsets(self, objective, constraints, nonlinear=None):
        """Maps each variable to a horizontal offset.

        Parameters
        ----------
        objective : Expression
            The canonicalized objective.
        constraints : list
            The canonicalized constraints.
        nonlinear : list, optional
            Non-linear constraints for CVXOPT.

        Returns
        -------
        tuple
            (map of variable to offset, length of variable vector)
        """
        vars_ = lu.get_expr_vars(objective)
        for constr in constraints:
            vars_ += lu.get_expr_vars(constr.expr)

        # If CVXOPT is the solver, some of the variables are
        # in NonLinearConstraints.
        if nonlinear is not None:
            for constr in nonlinear:
                for nonlin_var in constr.variables():
                    vars_ += lu.get_expr_vars(nonlin_var)

        var_offsets = OrderedDict()
        var_sizes = {}
        # Ensure the variables are always in the same
        # order for the same problem.
        var_names = list(set(vars_))
        var_names.sort(key=lambda (var_id, var_size): var_id)
        # Map var ids to offsets.
        vert_offset = 0
        for var_id, var_size in var_names:
            var_sizes[var_id] = var_size
            var_offsets[var_id] = vert_offset
            vert_offset += var_size[0]*var_size[1]

        return (var_offsets, var_sizes, vert_offset)

    def _save_dual_values(self, result_vec, constraints, constr_type):
        """Saves the values of the dual variables.

        Parameters
        ----------
        results_vec : array_like
            A vector containing the dual variable values.
        constraints : list
            A list of the LinEqConstr/LinLeqConstr in the problem.
        constr_type : type
            EqConstr or LeqConstr
        """
        constr_offsets = {}
        offset = 0
        for constr in constraints:
            constr_offsets[constr.constr_id] = offset
            offset += constr.size[0]*constr.size[1]
        active_constraints = []
        for constr in self.constraints:
            # Ignore constraints of the wrong type.
            if type(constr) == constr_type:
                active_constraints.append(constr)
        self._save_values(result_vec, active_constraints, constr_offsets)

    def _save_values(self, result_vec, objects, offset_map):
        """Saves the values of the optimal primal/dual variables.

        Parameters
        ----------
        results_vec : array_like
            A vector containing the variable values.
        objects : list
            The variables or constraints where the values will be stored.
        offset_map : dict
            A map of object id to offset in the results vector.
        """
        if len(result_vec) > 0:
            # Cast to desired matrix type.
            result_vec = self._DENSE_INTF.const_to_matrix(result_vec)
        for obj in objects:
            rows, cols = obj.size
            if obj.id in offset_map:
                offset = offset_map[obj.id]
                # Handle scalars
                if (rows, cols) == (1,1):
                    value = intf.index(result_vec, (offset, 0))
                else:
                    value = self._DENSE_INTF.zeros(rows, cols)
                    self._DENSE_INTF.block_add(value,
                        result_vec[offset:offset + rows*cols],
                        0, 0, rows, cols)
                offset += rows*cols
            else: # The variable was multiplied by zero.
                value = self._DENSE_INTF.zeros(rows, cols)
            obj.save_value(value)

    def _get_obj(self, objective, var_offsets, x_length,
                 matrix_intf, vec_intf):
        """Wraps _constr_matrix so it can be called for the objective.
        """
        dummy_constr = lu.create_eq(objective)
        return self._constr_matrix([dummy_constr], var_offsets, x_length,
                                   matrix_intf, vec_intf)

    def _constr_matrix(self, constraints, var_offsets, x_length,
                       matrix_intf, vec_intf):
        """Returns a matrix and vector representing a list of constraints.

        In the matrix, each constraint is given a block of rows.
        Each variable coefficient is inserted as a block with upper
        left corner at matrix[variable offset, constraint offset].
        The constant term in the constraint is added to the vector.

        Parameters
        ----------
        constraints : list
            A list of constraints.
        var_offsets : dict
            A dict of variable id to horizontal offset.
        x_length : int
            The length of the x vector.
        matrix_intf : interface
            The matrix interface to use for creating the constraints matrix.
        vec_intf : interface
            The matrix interface to use for creating the constant vector.

        Returns
        -------
        tuple
            A (matrix, vector) tuple.
        """

        rows = sum([c.size[0] * c.size[1] for c in constraints])
        cols = x_length
        V, I, J = [], [], []
        const_vec = vec_intf.zeros(rows, 1)
        vert_offset = 0
        for constr in constraints:
            coeffs = op2mat.get_coefficients(constr.expr)
            for id_, block in coeffs:
                vert_start = vert_offset
                vert_end = vert_start + constr.size[0]*constr.size[1]
                if id_ is lo.CONSTANT_ID:
                    # Flatten the block.
                    block = self._DENSE_INTF.const_to_matrix(block)
                    block_size = intf.size(block)
                    block = self._DENSE_INTF.reshape(
                        block,
                        (block_size[0]*block_size[1], 1)
                    )
                    const_vec[vert_start:vert_end, :] += block
                else:
                    horiz_offset = var_offsets[id_]
                    if intf.is_scalar(block):
                        block = intf.scalar_value(block)
                        V.append(block)
                        I.append(vert_start)
                        J.append(horiz_offset)
                    else:
                        # Block is a numpy matrix or
                        # scipy CSC sparse matrix.
                        if not intf.is_sparse(block):
                            block = self._SPARSE_INTF.const_to_matrix(block)
                        block = block.tocoo()
                        V.extend(block.data)
                        I.extend(block.row + vert_start)
                        J.extend(block.col + horiz_offset)
            vert_offset += constr.size[0]*constr.size[1]

        # Create the constraints matrix.
        if len(V) > 0:
            matrix = sp.coo_matrix((V, (I, J)), (rows, cols))
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

        big_x = self._CVXOPT_DENSE_INTF.zeros(cols, 1)
        for constr in nl_constr:
            constr.place_x0(big_x, var_offsets, self._CVXOPT_DENSE_INTF)

        def F(x=None, z=None):
            if x is None:
                return rows, big_x
            big_f = self._CVXOPT_DENSE_INTF.zeros(rows, 1)
            big_Df = self._CVXOPT_SPARSE_INTF.zeros(rows, cols)
            if z:
                big_H = self._CVXOPT_SPARSE_INTF.zeros(cols, cols)

            offset = 0
            for constr in nl_constr:
                constr_entries = constr.size[0]*constr.size[1]
                local_x = constr.extract_variables(x, var_offsets,
                                                   self._CVXOPT_DENSE_INTF)
                if z:
                    f, Df, H = constr.f(local_x,
                                        z[offset:offset + constr_entries])
                else:
                    result = constr.f(local_x)
                    if result:
                        f, Df = result
                    else:
                        return None
                big_f[offset:offset + constr_entries] = f
                constr.place_Df(big_Df, Df, var_offsets,
                                offset, self._CVXOPT_SPARSE_INTF)
                if z:
                    constr.place_H(big_H, H, var_offsets,
                                   self._CVXOPT_SPARSE_INTF)
                offset += constr_entries

            if z is None:
                return big_f, big_Df
            return big_f, big_Df, big_H
        return F

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Problem(%s, %s)" % (repr(self.objective),
                                    repr(self.constraints))
