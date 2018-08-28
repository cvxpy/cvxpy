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
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp
import numpy as np
from cvxpy.cvxcore.python import canonInterface


class MatrixCache(object):
    """A cached version of the matrix and vector pair in an affine constraint.

    Attributes
    ----------
    coo_tup : tuple
            A (V, I, J) triplet for the matrix.
    param_coo_tup : tuple
            A (V, I, J) triplet for the parameterized matrix.
    const_vec : array
        The vector offset.
    constraints : list
        A list of constraints in the matrix.
    shape : tuple
        The (rows, cols) dimensions of the matrix.
    """

    def __init__(self, coo_tup, const_vec, constraints, x_length):
        self.coo_tup = coo_tup
        self.const_vec = const_vec
        self.constraints = constraints
        rows = sum(np.prod(c.shape, dtype=int) for c in constraints)
        cols = x_length
        self.shape = (rows, cols)
        self.param_coo_tup = ([], [], [])

    def reset_param_data(self):
        """Clear old parameter data.
        """
        self.param_coo_tup = ([], [], [])


class MatrixData(object):
    """The matrices for the conic form convex optimization problem.

    Attributes
    ----------
    sym_data : SymData object
        The symbolic data for the conic form problem.
    matrix_intf : interface
        The matrix interface to use for creating the constraints matrix.
    vec_intf : interface
        The matrix interface to use for creating the constant vector.
    nonlin : bool
        Are nonlinear constraints needed?
    """

    def __init__(self, sym_data, matrix_intf, vec_intf, solver, nonlin):
        self.sym_data = sym_data
        # A dummy constraint for the objective.
        self.matrix_intf = matrix_intf
        self.vec_intf = vec_intf

        # Cache everything possible.
        self.obj_cache = self._init_matrix_cache(self._dummy_constr(),
                                                 self.sym_data.x_length)
        self._lin_matrix(self.obj_cache, caching=True)
        # Separate constraints based on the solver being used.
        constr_types = solver.split_constr(self.sym_data.constr_map)
        eq_constr, ineq_constr, nonlin_constr = constr_types
        # Equaliy constraints.
        self.eq_cache = self._init_matrix_cache(eq_constr,
                                                self.sym_data.x_length)
        self._lin_matrix(self.eq_cache, caching=True)
        # Inequality constraints.
        self.ineq_cache = self._init_matrix_cache(ineq_constr,
                                                  self.sym_data.x_length)
        self._lin_matrix(self.ineq_cache, caching=True)
        # Nonlinear constraints.
        if nonlin:
            self.F = self._nonlin_matrix(nonlin_constr)
        else:
            self.F = None

    def _dummy_constr(self):
        """Returns a dummy constraint for the objective.
        """
        return [lu.create_eq(self.sym_data.objective)]

    def get_objective(self):
        """Returns the linear objective and a scalar offset.
        """
        c, offset = self._cache_to_matrix(self.obj_cache)
        c = self.vec_intf.const_to_matrix(c.T, convert_scalars=True)
        c = intf.from_2D_to_1D(c)
        offset = self.vec_intf.scalar_value(offset)
        # Negate offset because was negated before.
        return c, -offset

    def get_eq_constr(self):
        """Returns the matrix and vector for the equality constraint.
        """
        return self._cache_to_matrix(self.eq_cache)

    def get_ineq_constr(self):
        """Returns the matrix and vector for the inequality constraint.
        """
        return self._cache_to_matrix(self.ineq_cache)

    def get_nonlin_constr(self):
        """Returns the oracle function for the nonlinear constraints.
        """
        return self.F

    def _init_matrix_cache(self, constraints, x_length):
        """Initializes the data structures for the cached matrix.

        Parameters
        ----------
        constraints : list
            A list of constraints in the matrix.
        x_length : int
            The number of columns in the matrix.
        Returns
        -------
        ((V, I, J), array)
        """
        rows = sum(np.prod(c.shape, dtype=int) for c in constraints)
        COO = ([], [], [])
        const_vec = self.vec_intf.zeros((rows, 1))
        return MatrixCache(COO, const_vec, constraints, x_length)

    def _lin_matrix(self, mat_cache, caching=False):
        """Computes a matrix and vector representing a list of constraints.

        In the matrix, each constraint is given a block of rows.
        Each variable coefficient is inserted as a block with upper
        left corner at matrix[variable offset, constraint offset].
        The constant term in the constraint is added to the vector.

        Parameters
        ----------
        mat_cache : MatrixCache
            The cached version of the matrix-vector pair.
        caching : bool
            Is the data being cached?
        """
        active_constr = []
        constr_offsets = []
        vert_offset = 0
        for constr in mat_cache.constraints:
            # Process the constraint if it has a parameter and not caching
            # or it doesn't have a parameter and caching.
            has_param = len(lu.get_expr_params(constr.expr)) > 0
            if (has_param and not caching) or (not has_param and caching):
                # If parameterized, convert the parameters into constant nodes.
                if has_param:
                    constr = lu.copy_constr(constr,
                                            lu.replace_params_with_consts)
                active_constr.append(constr)
                constr_offsets.append(vert_offset)
            vert_offset += np.prod(constr.shape, dtype=int)
        # Convert the constraints into a matrix and vector offset
        # and add them to the matrix cache.
        if len(active_constr) > 0:
            V, I, J, const_vec = canonInterface.get_problem_matrix(
                active_constr,
                self.sym_data.var_offsets,
                constr_offsets
            )
            # Convert the constant offset to the correct data type.
            conv_vec = self.vec_intf.const_to_matrix(const_vec,
                                                     convert_scalars=True)
            mat_cache.const_vec[:const_vec.size] += conv_vec
            for i, vals in enumerate([V, I, J]):
                mat_cache.coo_tup[i].extend(vals)

    def _cache_to_matrix(self, mat_cache):
        """Converts the cached representation of the constraints matrix.

        Parameters
        ----------
        mat_cache : MatrixCache
            The cached version of the matrix-vector pair.

        Returns
        -------
        A (matrix, vector) tuple.
        """
        # Get parameter values.
        param_cache = self._init_matrix_cache(mat_cache.constraints,
                                              mat_cache.shape[0])
        self._lin_matrix(param_cache)
        rows, cols = mat_cache.shape
        # Create the constraints matrix.
        # Combine the cached data with the parameter data.
        V, I, J = mat_cache.coo_tup
        Vp, Ip, Jp = param_cache.coo_tup
        if len(V) + len(Vp) > 0:
            matrix = sp.coo_matrix((V + Vp, (I + Ip, J + Jp)), (rows, cols))
            # Convert the constraints matrix to the correct type.
            matrix = self.matrix_intf.const_to_matrix(matrix,
                                                      convert_scalars=True)
        else:  # Empty matrix.
            matrix = self.matrix_intf.zeros(rows, cols)
        # Convert 2D ND arrays to 1D
        combo_vec = mat_cache.const_vec + param_cache.const_vec
        const_vec = intf.from_2D_to_1D(combo_vec)
        return (matrix, -const_vec)

    def _nonlin_matrix(self, nonlin_constr):
        """Returns an oracle for the nonlinear constraints.

        The oracle computes the combined function value, gradient, and Hessian.

        Parameters
        ----------
        nonlin_constr : list
            A list of nonlinear constraints represented as oracle functions.

        Returns
        -------
        Oracle function.
        """
        import cvxopt
        rows = int(sum(np.prod(c.shape, dtype=int) for c in nonlin_constr))
        cols = int(self.sym_data.x_length)
        var_offsets = self.sym_data.var_offsets

        big_x = cvxopt.matrix(0., (cols, 1))
        for constr in nonlin_constr:
            constr.place_x0(big_x, var_offsets)

        def F(x=None, z=None):
            """Oracle for function value, gradient, and Hessian.
            """
            if x is None:
                return rows, big_x
            big_f = cvxopt.matrix(0., (rows, 1))
            big_Df = cvxopt.spmatrix(0., [], [], size=(rows, cols))
            if z:
                big_H = cvxopt.spmatrix(0., [], [], size=(cols, cols))
            offset = 0
            for constr in nonlin_constr:
                constr_entries = np.prod(constr.shape, dtype=int)
                local_x = constr.extract_variables(x, var_offsets)
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
                constr.place_Df(big_Df, Df, var_offsets, offset)
                if z:
                    constr.place_H(big_H, H, var_offsets)
                offset += constr_entries

            if z is None:
                return big_f, big_Df
            return big_f, big_Df, big_H
        return F
