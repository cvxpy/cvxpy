"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.problems.solvers.solver import Solver
import numpy as np


class Elemental(Solver):
    """An interface for the Elemental solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = False

    # Map of Elemental status to CVXPY status.
    # TODO
    STATUS_MAP = {0: s.OPTIMAL}

    def import_solver(self):
        """Imports the solver.
        """
        import El
        El  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.ELEMENTAL

    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        return intf.DEFAULT_SPARSE_INTF

    def vec_intf(self):
        """The interface for vectors passed to the solver.
        """
        return intf.DEFAULT_INTF

    def split_constr(self, constr_map):
        """Extracts the equality, inequality, and nonlinear constraints.

        Parameters
        ----------
        constr_map : dict
            A dict of the canonicalized constraints.

        Returns
        -------
        tuple
            (eq_constr, ineq_constr, nonlin_constr)
        """
        return (constr_map[s.EQ], constr_map[s.LEQ], [])

    @staticmethod
    def distr_vec(local_vec, tag):
        """Converts the given vector to a distributed multivector.

        Parameters
        ----------
        local_vec : NumPy 1D array.
        tag : The Elemental data type.

        Returns
        -------
        Elemental distributed multivector.
        """
        import El
        vec = El.DistMultiVec(tag=tag)
        vec.Resize(local_vec.size, 1)
        for i in range(local_vec.size):
            vec.Set(i, 0, local_vec[i])
        return vec

    @staticmethod
    def local_vec(distr_vec):
        """Converts the given distributed multivector to a 1D array.

        Parameters
        ----------
        distr_vec : Elemental distributed multivector.

        Returns
        -------
        NumPy 1D array.
        """
        height = distr_vec.Height()
        local_vec = np.zeros(height)
        for i in range(local_vec.size):
            local_vec[i] = distr_vec.Get(i, 0)
        return local_vec

    @staticmethod
    def distr_mat(local_mat):
        """Converts the given matrix to a distributed sparse matrix.

        Parameters
        ----------
        local_mat : NumPy 2D array.

        Returns
        -------
        Elemental distributed sparse matrix.
        """
        import El
        local_mat = local_mat.tocoo()
        mat = El.DistSparseMatrix()
        mat.Resize(*local_mat.shape)
        mat.Reserve(len(local_mat.data))
        for val, i, j in zip(local_mat.data,
                             local_mat.row.astype(int),
                             local_mat.col.astype(int)):
            mat.QueueUpdate(i, j, val)
        mat.ProcessQueues()
        return mat

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
        import El
        data = self.get_problem_data(objective, constraints, cached_data)
        El.Initialize()
        # Package data.
        c = self.distr_vec(data["c"], El.dTag)
        A = self.distr_mat(data["A"])
        b = self.distr_vec(data["b"], El.dTag)
        G = self.distr_mat(data["G"])
        h = self.distr_vec(data["h"], El.dTag)
        dims = data["dims"]

        # Cone information.
        offset = 0
        orders = []
        firstInds = []
        for i in range(dims[s.LEQ_DIM]):
            orders.append(1)
            firstInds.append(offset)
            offset += 1
        for cone_len in dims[s.SOC_DIM]:
            for i in range(cone_len):
                orders.append(cone_len)
                firstInds.append(offset)
            offset += cone_len

        orders = self.distr_vec(np.array(orders), El.iTag)
        firstInds = self.distr_vec(np.array(firstInds), El.iTag)

        # Initialize empty vectors for solutions.
        x = El.DistMultiVec()
        y = El.DistMultiVec()
        z = El.DistMultiVec()
        s_var = El.DistMultiVec()
        if verbose:
            ctrl = El.SOCPAffineCtrl_d()
            ctrl.mehrotraCtrl.progress = True
            ctrl.mehrotraCtrl.time = True
        else:
            ctrl = None
        El.SOCPAffine(A, G, b, c, h, orders, firstInds, x, y, z, s_var, ctrl)
        local_c = data['c']
        local_x = self.local_vec(x)
        local_y = self.local_vec(y)
        local_z = self.local_vec(z)
        El.Finalize()
        results_dict = {'info': {'exitFlag': 0, 'pcost': local_c.dot(local_x)},
                        'x': local_x, 'y': local_y, 'z': local_z}
        return self.format_results(results_dict, data, cached_data)

    def format_results(self, results_dict, data, cached_data):
        """Converts the solver output into standard form.

        Parameters
        ----------
        results_dict : dict
            The solver output.
        data : dict
            Information about the problem.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The solver output in standard form.
        """
        new_results = {}
        status = self.STATUS_MAP[results_dict['info']['exitFlag']]
        new_results[s.STATUS] = status
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict['info']['pcost']
            new_results[s.VALUE] = primal_val + data[s.OFFSET]
            new_results[s.PRIMAL] = results_dict['x']
            new_results[s.EQ_DUAL] = results_dict['y']
            new_results[s.INEQ_DUAL] = results_dict['z']

        return new_results
