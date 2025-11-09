"""
Copyright, the CVXPY authors

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

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint


class broadcast_to(AffAtom):
    """Broadcast the expression given a shape input"""

    def __init__(self, expr, shape) -> None:
        self.broadcast_shape = shape
        self._shape = expr.shape
        self.broadcast_type = None
        super(broadcast_to, self).__init__(expr)

    def _supports_cpp(self) -> bool:
        return False

    def is_atom_log_log_convex(self) -> bool:
        return True

    def is_atom_log_log_concave(self) -> bool:
        return True

    def numeric(self, values):
        return np.broadcast_to(values[0], shape=self.broadcast_shape)

    def get_data(self) -> List[Optional[int]]:
        return [self.broadcast_shape]

    def validate_arguments(self) -> None:
        np.broadcast_to(
            np.empty(self.shape, dtype=np.dtype([])),
            shape=self.broadcast_shape
        )

    def shape_from_args(self) -> Tuple[int, ...]:
        return self.broadcast_shape

    def graph_implementation(
        self,
        arg_objs,
        shape: Tuple[int, ...],
        data=None,
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Broadcast an expression to a given shape.

        Parameters
        ----------
        arg_objs : list
            LinOp for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom. In this case data wraps axis

        Returns
        -------
        tuple
            (LinOp for the objective, list of constraints)
        """
        return (lu.broadcast_to(arg_objs, shape), [])
    
    def _verify_jacobian_args(self):
        """ We only support broadcasting to 2D arrays for now. """

        if self.broadcast_type:
            return True 
        else:
            if len(self.broadcast_shape) != 2:
                return False

            # cache the type of broadcasting
            m, n = self.broadcast_shape
            x = self.args[0]
            x_shape = tuple(x.shape)
            x_shape = (1,) * (2 - len(x_shape)) + x_shape
           
            # row-wise stacking
            if x_shape[0] == 1 and x_shape[1] == n:
                self.broadcast_type = "row"
            # column-wise stacking
            elif x_shape[0] == m and x_shape[1] == 1:
                self.broadcast_type = "col"
            # scalar to matrix
            if np.all(np.array(x_shape) == 1):
                self.broadcast_type = "scalar"

            return True

    def _jacobian(self):
        """
        We only support 2D arrays for now. There are three different types of broadcasting:

        1. Broadcasting a (1, n) array to (m, n)
        2. Broadcasting a (m, 1) array to (m, n)
        3. Broadcasting a (1, 1) array to (m, n)
        
        Mathematically, the three cases correspond to the following operations:

        1. Let phi(x) be the original array as a column vector of size (n, 1).
           The broadcasted array is equivalent to the multiplication  ones_m phi(x)^T.
        2. Let phi(x) be the original array as a column vector of size (m, 1).
           The broadcasted array is equivalent to the multiplication phi(x) ones_n^T.
        3. Let phi(x) be the original array as a scalar.
           The broadcasted array is equivalent to the multiplication 
           phi(x) ones_m ones_n^T.
         
        In all three cases, we can use the properties of the Kronecker product to compute
        the Jacobian of the broadcasted array with respect to x. Let vec(;) denote the 
        vectorization operator that stacks the columns of a matrix into a single column 
        vector. Then, we have:

        1. vec(ones_m phi(x)^T)        = A phi(x) where A = Kron(I_n, ones_m)
        2. vec(phi(x) ones_n^T)        = A phi(x) where A = Kron(ones_n, I_m)
        3. vec(phi(x) ones_m ones_n^T) = phi(x) ones_mn
        
        The Jacobians can then be computed as follows:

        1. J_broadcast = A J_phi
        2. J_broadcast = A J_phi
        3. J_broadcast = ones_mn dphi/dx

        Note that multiplication by A can be implemented efficiently without forming A explicitly.
        """
        m, n = self.broadcast_shape
       
        if self.broadcast_type == "row":
            jac_x_dict = self.args[0].jacobian()
            for key in jac_x_dict:
                rows, cols, vals = jac_x_dict[key]
                rows = np.repeat(rows * m, m) + np.tile(np.arange(m), len(rows))
                cols = np.repeat(cols, m)
                vals = np.repeat(vals, m)
                jac_x_dict[key] = (rows, cols, vals)
            return jac_x_dict
        elif self.broadcast_type == "col":
            jac_x_dict = self.args[0].jacobian()
            for key in jac_x_dict:
                rows, cols, vals = jac_x_dict[key]
                rows = np.repeat(rows, n) + np.tile(np.arange(n) * m, len(rows))
                cols = np.repeat(cols, n)
                vals = np.repeat(vals, n)
                jac_x_dict[key] = (rows, cols, vals)
            return jac_x_dict
        elif self.broadcast_type == "scalar":
            jac_x_dict = self.args[0].jacobian()
            for key in jac_x_dict:
                rows, cols, vals = jac_x_dict[key]
                rows = np.tile(np.arange(m * n), len(rows))
                cols = np.repeat(cols, m * n)
                vals = np.repeat(vals, m * n)
                jac_x_dict[key] = (rows, cols, vals)
            return jac_x_dict
        else:
            raise NotImplementedError("Jacobian not implemented for broadcast_to.")

    def _verify_hess_vec_args(self):
        return self._verify_jacobian_args()
    
    def _hess_vec(self, vec):
        x = self.args[0]
        m, n = self.broadcast_shape
        
        if self.broadcast_type == "row":
            return x._hess_vec(vec.reshape(n, m).sum(axis=1))
        elif self.broadcast_type == "col":
            return x._hess_vec(vec.reshape(n, m).sum(axis=0))
        elif self.broadcast_type == "scalar":
            return x._hess_vec(vec.sum())
        else:
            raise NotImplementedError("hess-vec not implemented for broadcast_to.")