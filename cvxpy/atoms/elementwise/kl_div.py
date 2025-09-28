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
from scipy.sparse import csc_array
from scipy.special import kl_div as kl_div_scipy

from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable


class kl_div(Elementwise):
    """:math:`x\\log(x/y) - x + y`

    For disambiguation between kl_div and rel_entr, see https://github.com/cvxpy/cvxpy/issues/733
    """

    def __init__(self, x, y) -> None:
        super(kl_div, self).__init__(x, y)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        x = values[0]
        y = values[1]
        return kl_div_scipy(x, y)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values) -> List[Optional[csc_array]]:
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if np.min(values[0]) <= 0 or np.min(values[1]) <= 0:
            # Non-differentiable.
            return [None, None]
        else:
            div = values[0]/values[1]
            grad_vals = [np.log(div), 1 - div]
            grad_list = []
            for idx in range(len(values)):
                rows = self.args[idx].size
                cols = self.size
                grad_list += [kl_div.elemwise_grad_to_diag(grad_vals[idx],
                                                           rows, cols)]
            return grad_list

    def _verify_hess_vec_args(self):
        x = self.args[0]
        y = self.args[1]
        
        # we check that the arguments are of the same size or one of them 
        # is a scalar
        if not (x.size == 1 or y.size == 1 or x.size == y.size):
            return False

        # we assume both arguments must be variables (the case where one 
        # argument is constant should perhaps been caught in the canonicalization?)
        if not (isinstance(x, Variable) and isinstance(y, Variable)):
            return False

        # we assume that the arguments correspond to different variables
        # (otherwise the differentation logic fails)
        if x.id == y.id:
            return False 

        return True

    def _hess_vec(self, vec):
        """ See the docstring of the hess_vec method of the atom class. """
        x = self.args[0]
        y = self.args[1]
        dx2_vals = vec / x.value
        dy2_vals = vec * x.value / (y.value ** 2)
        dxdy_vals = - vec / y.value

        if x.size == 1:
            idxs = np.arange(y.size)
            zeros_y = np.zeros(y.size, dtype=int)
            return {(x, x): (0, 0, np.sum(dx2_vals)),
                    (y, y): (idxs, idxs, dy2_vals),
                    (x, y): (zeros_y, idxs, dxdy_vals),
                    (y, x): (idxs, zeros_y, dxdy_vals)}
        elif y.size == 1:
            idxs = np.arange(x.size)
            zeros_x = np.zeros(x.size, dtype=int)
            return {(x, x): (idxs, idxs, dx2_vals), 
                    (y, y): (0, 0, np.sum(dy2_vals)),
                    (x, y): (idxs, zeros_x, dxdy_vals),
                    (y, x): (zeros_x, idxs, dxdy_vals)}
        else:
            idxs = np.arange(x.size)
            return {(x, x): (idxs, idxs, dx2_vals), 
                    (y, y): (idxs, idxs, dy2_vals),
                    (x, y): (idxs, idxs, dxdy_vals),
                    (y, x): (idxs, idxs, dxdy_vals)}
    
    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= 0, self.args[1] >= 0]
    
    def point_in_domain(self, argument=0):
        return np.ones(self.args[argument].shape)
