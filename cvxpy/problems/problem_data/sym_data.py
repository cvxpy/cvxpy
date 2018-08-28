"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""
from cvxpy.cvxcore.python import canonInterface
import cvxpy.settings as s
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.lin_ops as lo
from cvxpy.constraints import SOC, PSD, ExpCone
from toolz.itertoolz import unique
from collections import OrderedDict
import numpy as np


class SymData(object):
    """The symbolic info for the conic form convex optimization problem.

    Attributes
    ----------
    objective : LinOp
        The linear operator representing the objective.
    constraints : list
        The list of canonicalized constraints.
    constr_map : dict
        A map of constraint type to list of constraints.
    dims : dict
        The dimensions of the cones.
    var_offsets : dict
        A dict of variable id to horizontal offset.
    var_shapes : dict
        A dict of variable id to variable dimensions.
    x_length : int
        The length of the x vector.
    presolve_status : str or None
        The status returned by the presolver.
    """

    def __init__(self, objective, constraints, solver):
        self.objective = objective
        self.constraints = constraints
        self.constr_map = self.filter_constraints(constraints)
        # TODO drop this because it must be inverted.
        # self.presolve_status = self.presolve(self.objective, self.constr_map)
        self.presolve_status = None
        self.dims = self.format_for_solver(self.constr_map, solver)

        all_ineq = self.constr_map[s.EQ] + self.constr_map[s.LEQ]
        # CVXOPT can have variables that only live in NonLinearConstraints.
        nonlinear = self.constr_map[s.EXP] if solver.name() == s.CVXOPT else []
        var_data = self.get_var_offsets(objective, all_ineq, nonlinear)
        self.var_offsets, self.var_shapes, self.x_length = var_data

    @staticmethod
    def filter_constraints(constraints):
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
                      s.PSD: [],
                      s.EXP: [],
                      s.BOOL: [],
                      s.INT: []}
        for c in constraints:
            if isinstance(c, lo.LinEqConstr):
                constr_map[s.EQ].append(c)
            elif isinstance(c, lo.LinLeqConstr):
                constr_map[s.LEQ].append(c)
            elif isinstance(c, SOC):
                constr_map[s.SOC].append(c)
            elif isinstance(c, PSD):
                constr_map[s.PSD].append(c)
            elif isinstance(c, ExpCone):
                constr_map[s.EXP].append(c)
        return constr_map

    @staticmethod
    def presolve(objective, constr_map):
        """Eliminates unnecessary constraints and short circuits the solver
        if possible.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constr_map : dict
            A map of constraint type to a list of constraints.

        Returns
        -------
        bool
            Is the problem infeasible?
        """
        # Remove redundant constraints.
        for key, constraints in constr_map.items():
            uniq_constr = unique(constraints,
                                 key=lambda c: c.constr_id)
            constr_map[key] = list(uniq_constr)

        # If there are no constraints, the problem is unbounded
        # if any of the coefficients are non-zero.
        # If all the coefficients are zero then return the constant term
        # and set all variables to 0.
        if not any(constr_map.values()):
            str(objective)  # TODO

        # Remove constraints with no variables or parameters.
        for key in [s.EQ, s.LEQ]:
            new_constraints = []
            for constr in constr_map[key]:
                vars_ = lu.get_expr_vars(constr.expr)
                if len(vars_) == 0 and not lu.get_expr_params(constr.expr):
                    V, I, J, coeff = canonInterface.get_problem_matrix([constr])
                    is_pos, is_neg = intf.sign(coeff)
                    # For equality constraint, coeff must be zero.
                    # For inequality (i.e. <= 0) constraint,
                    # coeff must be negative.
                    if key == s.EQ and not (is_pos and is_neg) or \
                            key == s.LEQ and not is_neg:
                        return s.INFEASIBLE
                else:
                    new_constraints.append(constr)
            constr_map[key] = new_constraints

        return None

    @staticmethod
    def format_for_solver(constr_map, solver):
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
        dims[s.EQ_DIM] = sum(np.prod(c.shape, dtype=int) for c in constr_map[s.EQ])
        dims[s.LEQ_DIM] = sum(np.prod(c.shape, dtype=int) for c in constr_map[s.LEQ])
        dims[s.SOC_DIM] = []
        dims[s.PSD_DIM] = []
        dims[s.EXP_DIM] = 0
        # Formats nonlinear constraints for the solver.
        for constr_type in constr_map.keys():
            if constr_type not in [s.EQ, s.LEQ]:
                for constr in constr_map[constr_type]:
                    constr.format(constr_map[s.EQ], constr_map[s.LEQ],
                                  dims, solver)

        return dims

    @staticmethod
    def get_var_offsets(objective, constraints, nonlinear):
        """Maps each variable to a horizontal offset.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The canonicalized constraints.
        nonlinear : list
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
        for constr in nonlinear:
            for nonlin_var in constr.args:
                vars_ += lu.get_expr_vars(nonlin_var)

        var_offsets = OrderedDict()
        # Ensure the variables are always in the same
        # order for the same problem.
        var_names = list(set(vars_))
        var_names.sort(key=lambda id_and_shape: id_and_shape[0])
        # Map var ids to offsets and shape.
        var_shapes = {}
        vert_offset = 0
        for var_id, var_shape in var_names:
            var_shapes[var_id] = var_shape
            var_offsets[var_id] = vert_offset
            vert_offset += np.prod(var_shape, dtype=int)

        return (var_offsets, var_shapes, vert_offset)
