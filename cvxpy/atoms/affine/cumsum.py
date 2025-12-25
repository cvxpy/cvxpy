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
"""
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable


def get_diff_mat(dim: int, axis: int) -> sp.csc_array:
    """Return a sparse matrix representation of first order difference operator.

    Parameters
    ----------
    dim : int
       The length of the matrix dimensions.
    axis : int
       The axis to take the difference along.

    Returns
    -------
    sp.csc_array
        A square matrix representing first order difference.
    """
    mat = sp.diags_array([np.ones(dim), -np.ones(dim - 1)], offsets=[0, -1],
                   shape=(dim, dim),
                   format='csc')
    return mat if axis == 0 else mat.T


def _flatten_c_order(linop: lo.LinOp, target_shape: Tuple[int, ...]) -> lo.LinOp:
    """Flatten a LinOp in C-order (row-major) for use in graph_implementation.

    CVXPY's lin_ops use F-order (column-major) internally. To achieve C-order
    flattening, we transpose (reverse all axes), then reshape. This is the same
    pattern used in reshape.graph_implementation for order='C'.

    For a 2D array [[a, b], [c, d]], F-order gives [a, c, b, d] while
    C-order gives [a, b, c, d]. The transpose swaps axes so that F-order
    reshape of the transposed array produces the C-order result.

    Parameters
    ----------
    linop : lo.LinOp
        The LinOp to flatten.
    target_shape : tuple
        The target shape after flattening (typically 1D).

    Returns
    -------
    lo.LinOp
        The flattened LinOp in C-order.
    """
    ndim = len(linop.shape)
    if ndim <= 1:
        return linop
    # Reverse axes: equivalent to multiple transposes that reverse memory layout
    perm = list(range(ndim))[::-1]
    transposed = lu.transpose(linop, perm)
    return lu.reshape(transposed, target_shape)


class cumsum(AffAtom, AxisAtom):
    """
    Cumulative sum of the elements of an expression.

    Attributes
    ----------
    expr : CVXPY expression
        The expression being summed.
    axis : int, optional
        The axis to sum across. If None, the array is flattened before cumsum.
        Note: NumPy's default is axis=None, while CVXPY defaults to axis=0.
    """
    def __init__(self, expr: Expression, axis: Optional[int] = 0) -> None:
        super(cumsum, self).__init__(expr, axis)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """
        Returns the cumulative sum of elements of an expression over an axis.
        """
        return np.cumsum(values[0], axis=self.axis)
    
    def shape_from_args(self) -> Tuple[int, ...]:
        """Flattened if axis=None, otherwise same as input."""
        if self.axis is None:
            return (self.args[0].size,)
        return self.args[0].shape

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        ndim = len(values[0].shape)
        axis = self.axis

        # Handle axis=None: treat as 1D cumsum over C-order flattened array
        if axis is None:
            dim = values[0].size
            # Lower triangular matrix = cumsum gradient in C-order space
            tril = sp.csc_array(np.tril(np.ones((dim, dim))))
            # Permutation to convert F-order vectorized input to C-order
            # P[i, j] = 1 means output[i] = input[j], so P @ f_vec = c_vec
            c_order_indices = np.arange(dim).reshape(values[0].shape, order='F').flatten(order='C')
            P = sp.csc_array((np.ones(dim), (np.arange(dim), c_order_indices)), shape=(dim, dim))
            # Gradient: input (F-order) -> C-order via P -> cumsum via tril -> output (1D)
            # dy = tril @ P @ dx_f, so gradient is tril @ P
            grad = tril @ P
            return [sp.csc_array(grad)]

        if axis < 0:
            axis = ndim + axis
        dim = values[0].shape[axis]

        # Lower triangular matrix = cumsum gradient
        tril = sp.csc_array(np.tril(np.ones((dim, dim))))

        if ndim <= 2:
            # Existing 2D logic
            var = Variable(self.args[0].shape)
            if axis == 0:
                grad = MulExpression(tril, var)._grad(values)[1]
            else:
                grad = MulExpression(var, tril.T)._grad(values)[0]
            return [grad]

        # ND: Kronecker product I_post ⊗ tril ⊗ I_pre
        pre_size = int(np.prod(values[0].shape[:axis])) if axis > 0 else 1
        post_size = int(np.prod(values[0].shape[axis+1:])) if axis < ndim - 1 else 1

        grad = sp.kron(sp.kron(sp.eye_array(post_size), tril), sp.eye_array(pre_size))
        return [sp.csc_array(grad)]

    def get_data(self):
        """Returns the axis being summed."""
        return [self.axis]

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Cumulative sum via difference matrix.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        # Implicit O(n) definition:
        # X = Y[1:,:] - Y[:-1, :]
        Y = lu.create_var(shape)
        axis = data[0]

        # Handle axis=None: flatten in C order, then 1D cumsum
        if axis is None:
            dim = int(np.prod(shape))  # shape is (total_size,) for axis=None
            diff_mat = get_diff_mat(dim, axis=0)
            diff_mat = lu.create_const(diff_mat, (dim, dim), sparse=True)

            flat_input = _flatten_c_order(arg_objs[0], shape)

            # Apply diff matrix: D @ Y = X_flat
            diff = lu.mul_expr(diff_mat, Y, shape)
            return (Y, [lu.create_eq(flat_input, diff)])

        ndim = len(shape)

        # Normalize negative axis
        if axis < 0:
            axis = ndim + axis

        dim = shape[axis]

        # 1D/2D: use existing optimized path
        if ndim <= 2:
            diff_mat = get_diff_mat(dim, axis)
            diff_mat = lu.create_const(diff_mat, (dim, dim), sparse=True)
            if axis == 0:
                diff = lu.mul_expr(diff_mat, Y, shape)
            else:
                diff = lu.rmul_expr(Y, diff_mat, shape)
            return (Y, [lu.create_eq(arg_objs[0], diff)])

        # ND: transpose -> reshape -> mul -> reshape -> transpose
        # This reduces ND cumsum to 2D by bringing target axis to front.
        pre_axes = list(range(axis))
        post_axes = list(range(axis + 1, ndim))

        # Permutation: bring axis to front
        perm = [axis] + pre_axes + post_axes
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i

        transposed_shape = tuple(shape[p] for p in perm)

        # Flatten to 2D: (dim, other_size)
        other_size = int(np.prod(shape) // dim)
        flat_shape = (dim, other_size)

        # Diff matrix for axis 0
        diff_mat = get_diff_mat(dim, axis=0)
        diff_mat = lu.create_const(diff_mat, (dim, dim), sparse=True)

        # Apply: transpose -> reshape -> mul -> reshape -> transpose
        Y_t = lu.transpose(Y, perm)
        Y_flat = lu.reshape(Y_t, flat_shape)
        diff_flat = lu.mul_expr(diff_mat, Y_flat, flat_shape)
        diff_t = lu.reshape(diff_flat, transposed_shape)
        diff = lu.transpose(diff_t, inv_perm)

        return (Y, [lu.create_eq(arg_objs[0], diff)])
