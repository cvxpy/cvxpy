"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.atoms.atom import Atom
import cvxpy as cp
from typing import Tuple
import numpy as np

class perspective(Atom):
    """TODO.
    """

    def __init__(self, f: cp.Expression, x: cp.Expression , s: cp.Expression) -> None:
        super(perspective, self).__init__(f,x,s)

    def validate_arguments(self) -> None:
        assert self.args[0].size == 1 # dealing only with scalars, for now
        assert self.args[1].size == 1
        assert self.args[2].size == 1
        return super().validate_arguments()

    @Atom.numpy_numeric
    def numeric(self, values):
        """
        Compute the perspective sf(x/s) numerically.
        """

        assert values[1] >= 0

        x_val = np.array(values[0])
        s_val = np.array(values[1])
        f = self.args[0]
        
        rat = np.array([x_val/s_val])
        return np.array([f.numeric(rat)*s_val])

    def _grad(self, values):
        """
        """
        pass

    def _column_grad(self, value):
        pass

    def sign_from_args(self) -> Tuple[bool, bool]:
        f_pos = self.args[0].is_nonneg()
        f_neg = self.args[0].is_nonpos()
        s_pos = self.args[2].is_nonneg()
        
        assert s_pos

        is_positive = (f_pos and s_pos)
        is_negative = (f_neg and s_pos)

        return is_positive, is_negative

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return self.args[0].is_convex() and self.args[2].is_nonneg()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return self.args[0].is_concave() and self.args[2].is_nonneg()

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        assert idx in [1,2] "can't handle increasing in 'f'"
        if idx == 1:
            return self.args[0].is_incr(0) # assuming scalar for now
        elif idx == 2:
            return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        assert idx in [1,2]
        if idx == 1:
            return self.args[0].is_decr(0) # assuming scalar for now
        elif idx == 2:
            return True
        pass
    
    def shape_from_args(self) -> Tuple[int, ...]:
            """Returns the (row, col) shape of the expression.
            """
            return self.args[0].shape

