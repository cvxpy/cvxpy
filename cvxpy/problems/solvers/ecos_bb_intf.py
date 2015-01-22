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
from cvxpy.problems.solvers.ecos_intf import ECOS
import ecos

class ECOS_BB(ECOS):
    """An interface for the ECOS BB solver.
    """
    def name(self):
        """The name of the solver.
        """
        return s.ECOS_BB

    @staticmethod
    def _noncvx_id_to_idx(dims, var_offsets, var_sizes):
        """Converts the nonconvex constraint variable ids in dims into indices.

        Parameters
        ----------
        dims : dict
            The dimensions of the cones.
        var_offsets : dict
            A dict of variable id to horizontal offset.
        var_sizes : dict
            A dict of variable id to variable dimensions.

        Returns
        -------
        tuple
            A list of indices for the boolean variables and integer variables.
        """
        bool_idx = []
        int_idx = []
        for indices, constr_type in zip([bool_idx, int_idx],
                                        [s.BOOL_IDS, s.INT_IDS]):
            for var_id in dims[constr_type]:
                offset = var_offsets[var_id]
                size = var_sizes[var_id]
                for i in range(size[0]*size[1]):
                    indices.append(offset + i)
            del dims[constr_type]

        return bool_idx, int_idx

    def get_problem_data(self, objective, constraints, cached_data):
        """Returns the argument for the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The arguments needed for the solver.
        """
        data = super(ECOS_BB, self).get_problem_data(objective, constraints,
                                                     cached_data)
        sym_data = self.get_sym_data(objective, constraints, cached_data)
        bool_idx, int_idx = self._noncvx_id_to_idx(data[s.DIMS],
                                                   sym_data.var_offsets,
                                                   sym_data.var_sizes)
        data[s.BOOL_IDX] = bool_idx
        data[s.INT_IDX] = int_idx
        return data

    def solve(self, objective, constraints, cached_data,
              warm_start, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        data = self.get_problem_data(objective, constraints, cached_data)
        # Default verbose to false for BB wrapper.
        mi_verbose = solver_opts.get('mi_verbose', False)
        results_dict = ecos.solve(data[s.C], data[s.G], data[s.H],
                                  data[s.DIMS], data[s.A], data[s.B],
                                  verbose=verbose,
                                  mi_verbose=mi_verbose,
                                  bool_vars_idx=data[s.BOOL_IDX],
                                  int_vars_idx=data[s.INT_IDX],
                                  **solver_opts)
        return self.format_results(results_dict, None,
                                   data[s.OFFSET], cached_data)
