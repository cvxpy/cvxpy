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

import cvxpy.lin_ops as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.lin_ops.lin_to_matrix as op2mat

class ProblemData(object):
    """The data for a convex optimization problem.

    Attributes
    ----------
    var_offsets : dict
        A dict of variable id to horizontal offset.
    x_length : int
        The length of the x vector.
    objective : LinOp
        The linear operator representing the objective.
    eq_constr : list
        The linear equality constraints.
    ineq_constr : list
        The linear inequality constraints.
    nonlin_constr : list
        The nonlinear inequality constraints.
    matrix_intf : interface
        The matrix interface to use for creating the constraints matrix.
    vec_intf : interface
        The matrix interface to use for creating the constant vector.
    """
    def __init__(self, var_offsets, x_length, objective,
                 eq_constr, ineq_constr, nonlin_constr,
                 matrix_intf, vec_intf):
        self.var_offsets = var_offsets
        self.x_length = x_length
        # Objective and a dummy constraint.
        self.objective = objective
        self._dummy_constr = [lu.create_eq(self.objective)]

        self.eq_constr = eq_constr
        self.ineq_constr = ineq_constr
        self.nonlin_constr = nonlin_constr
        self.matrix_intf = matrix_intf
        self.vec_intf = vec_intf
        # Cache everything possible.
        self._cache_all()

    def _cache_all(self):
        """Caches all the data possible.
        """
        self.c_COO, self.offset =  self._init_matrix_cache(self._dummy_constr)
        self._lin_matrix(self._dummy_constr, self.c_COO,
                         self.offset, caching=True)
        # Equaliy constraints.
        self.A_COO, self.b = self._init_matrix_cache(self.eq_constr)
        self._lin_matrix(self.eq_constr, self.A_COO, self.b, caching=True)
        # Inequality constraints.
        self.G_COO, self.h = self._init_matrix_cache(self.ineq_constr)
        self._lin_matrix(self.ineq_constr, self.G_COO, self.h, caching=True)
        # Nonlinear constraints.
        self.F = self._nonlin_matrix()

    def get_objective(self):
        """Returns the linear objective and a scalar offset.
        """
        c, offset = self._cache_to_matrix(self._dummy_constr, self.c_COO,
                                          self.offset)
        offset = intf.matrix_intf.scalar_value(offset)
        return c.T, offset

    def get_eq_constr(self):
        """Returns the matrix and vector for the equality constraint.
        """
        return self._cache_to_matrix(self.eq_constr, self.A_COO, self.b)

    def get_ineq_constr(self):
        """Returns the matrix and vector for the inequality constraint.
        """
        return self._cache_to_matrix(self.eq_constr, self.G_COO, self.h)


    def _constr_matrix_size(self, constraints):
        """Returns the dimensions of the constraint matrix.

        Parameters
        ----------
        constraints : list
            A list of constraints in the matrix.
        Returns
        -------
        (rows, cols)
        """
        rows = sum([c.size[0] * c.size[1] for c in constraints])
        cols = self.x_length
        return (rows, cols)

    def _init_matrix_cache(self, constraints):
        """Initializes the data structures for the cached matrix.

        Parameters
        ----------
        constraints : list
            A list of constraints in the matrix.
        Returns
        -------
        ((V, I, J), array)
        """
        rows = _constr_matrix_size(constraints)[0]
        COO = ([], [], [])
        const_vec = self.vec_intf.zeros(rows, 1)
        return (COO, const_vec)

    def _lin_matrix(self, constraints, COO, const_vec, caching=False):
        """Computes a matrix and vector representing a list of constraints.

        In the matrix, each constraint is given a block of rows.
        Each variable coefficient is inserted as a block with upper
        left corner at matrix[variable offset, constraint offset].
        The constant term in the constraint is added to the vector.

        Parameters
        ----------
        constraints : list
            A list of constraints in the matrix.
        COO : tuple
            A (V, I, J) triplet.
        const_vec : array
            The constant term.
        caching : bool
            Is the data being cached?

        Returns
        -------
        tuple
            A (matrix, vector) tuple.
        """
        V, I, J = COO
        vert_offset = 0
        for constr in constraints:
            # Process the constraint if it has a parameter and not caching
            # or it doesn't have a parameter and caching.
            if lu.get_expr_params(constr.expr) != caching:
                self._process_constr(constr, V, I, J, const_vec, vert_offset)
            vert_offset += constr.size[0]*constr.size[1]

    def _cache_to_matrix(self, constraints, COO, const_vec):
        """Converts the cached representation of the constraints matrix.

        Parameters
        ----------
        constraints : list
            A list of constraints in the matrix.
        COO : tuple
            A (V, I, J) triplet.
        const_vec : array
            The constant term.

        Returns
        -------
        A (matrix, vector) tuple.
        """
        rows, cols = self._constr_matrix_size(constraints)
        # Create the constraints matrix.
        V, I, J = COO
        if len(V) > 0:
            matrix = sp.coo_matrix((V, (I, J)), (rows, cols))
            # Convert the constraints matrix to the correct type.
            matrix = matrix_intf.const_to_matrix(matrix, convert_scalars=True)
        else: # Empty matrix.
            matrix = matrix_intf.zeros(rows, cols)
        # Convert 2D ND arrays to 1D
        if self.vec_intf is intf.DEFAULT_INTERFACE:
            const_vec = intf.from_2D_to_1D(const_vec)
        return (matrix, -const_vec)

    def _process_constr(self, constr, V, I, J, const_vec, vert_offset):
        """Extract the coefficients from a constraint.

        Parameters
        ----------
        constr : LinConstr
            The linear constraint to process.
        V : list
            A list of values in the COO sparse matrix.
        I : list
            A list of rows in the COO sparse matrix.
        J : list
            A list of columns in the COO sparse matrix.
        const_vec : array
            The constant vector.
        vert_offset : int
            The row offset of the constraint.
        """
        coeffs = op2mat.get_coefficients(constr.expr)
        for id_, block in coeffs:
            vert_start = vert_offset
            vert_end = vert_start + constr.size[0]*constr.size[1]
            if id_ is lo.CONSTANT_ID:
                # Flatten the block.
                block = self.vec_intf.const_to_matrix(block)
                block_size = intf.size(block)
                block = self.vec_intf.reshape(
                    block,
                    (block_size[0]*block_size[1], 1)
                )
                const_vec[vert_start:vert_end, :] += block
            else:
                horiz_offset = self.var_offsets[id_]
                if intf.is_scalar(block):
                    block = intf.scalar_value(block)
                    V.append(block)
                    I.append(vert_start)
                    J.append(horiz_offset)
                else:
                    # Block is a numpy matrix or
                    # scipy CSC sparse matrix.
                    if not intf.is_sparse(block):
                        block = intf.DEFAULT_SPARSE_INTF.const_to_matrix(
                                    block
                                )
                    block = block.tocoo()
                    V.extend(block.data)
                    I.extend(block.row + vert_start)
                    J.extend(block.col + horiz_offset)

    def _nonlin_matrix(self):
        """Returns an oracle for the nonlinear constraints.

        The oracle computes the combined function value, gradient, and Hessian.

        Returns
        -------
        Oracle function.
        """
        rows = sum([c.size[0] * c.size[1] for c in self.nl_constr])
        cols = self.x_length

        big_x = self.vec_intf.zeros(cols, 1)
        for constr in self.nl_constr:
            constr.place_x0(big_x, self.var_offsets, self.vec_intf)

        def F(x=None, z=None):
            if x is None:
                return rows, big_x
            big_f = self.vec_intf.zeros(rows, 1)
            big_Df = self.matrix_intf.zeros(rows, cols)
            if z:
                big_H = self.matrix_intf.zeros(cols, cols)

            offset = 0
            for constr in self.nl_constr:
                constr_entries = constr.size[0]*constr.size[1]
                local_x = constr.extract_variables(x, self.var_offsets,
                                                   self.vec_intf)
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
                constr.place_Df(big_Df, Df, self.var_offsets,
                                offset, self.matrix_intf)
                if z:
                    constr.place_H(big_H, H, self.var_offsets,
                                   self.matrix_intf)
                offset += constr_entries

            if z is None:
                return big_f, big_Df
            return big_f, big_Df, big_H
        return F
