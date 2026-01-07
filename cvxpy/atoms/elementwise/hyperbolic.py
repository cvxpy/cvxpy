"""
Copyright 2025 CVXPY Developers

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
from typing import List, Tuple

import numpy as np

from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable


class sinh(Elementwise):
    """Elementwise :math:`\\sinh x`.
    """

    def __init__(self, x) -> None:
        super(sinh, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise sinh of x.
        """
        return np.sinh(values[0])
    
    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always unknown.
        raise NotImplementedError("sign_from_args not implemented for sinh.")

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_esr(self) -> bool:
        """Is the atom esr?
        """
        return True

    def is_atom_hsr(self) -> bool:
        """Is the atom hsr?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return []
    
    def _grad(self, values) -> List[Constraint]:
        raise NotImplementedError("Gradient not implemented for sinh.")

    def _verify_hess_vec_args(self):
        return isinstance(self.args[0], Variable)

    def _hess_vec(self, vec):
        var = self.args[0]
        idxs = np.arange(var.size)
        vals = np.sinh(var.value.flatten(order='F')) * vec
        return {(var, var): (idxs, idxs, vals)}
    
    def _verify_jacobian_args(self):
        return isinstance(self.args[0], Variable)

    def _jacobian(self):
        x = self.args[0]
        idxs = np.arange(x.size, dtype=int)
        vals = np.cosh(x.value.flatten(order='F'))
        return {x: (idxs, idxs, vals)}
        
class tanh(Elementwise):
    """Elementwise :math:`\\tan x`.
    """

    def __init__(self, x) -> None:
        super(tanh, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise hyperbolic tangent of x.
        """
        return np.tanh(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always unknown.
        raise NotImplementedError("sign_from_args not implemented for tanh.")

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False
    
    def is_atom_esr(self) -> bool:
        """Is the atom esr?
        """
        return True

    def is_atom_hsr(self) -> bool:
        """Is the atom hsr?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return []

    def _grad(self, values) -> List[Constraint]:
        raise NotImplementedError("Gradient not implemented for tanh.")

    def _verify_hess_vec_args(self):
        return isinstance(self.args[0], Variable)

    def _hess_vec(self, vec):
        var = self.args[0]
        idxs = np.arange(var.size)
        vals = -2 * (np.tanh(var.value) / np.cosh(var.value) ** 2).flatten(order='F') * vec
        return {(var, var): (idxs, idxs, vals)}
    
    def _verify_jacobian_args(self):
        return isinstance(self.args[0], Variable)

    def _jacobian(self):
        x = self.args[0]
        idxs = np.arange(x.size, dtype=int)
        vals = 1 / np.cosh(x.value.flatten(order='F'))**2
        return {x: (idxs, idxs, vals.flatten(order='F'))}
    
class asinh(Elementwise):
    """Elementwise :math:`\\operatorname{asinh} x` (inverse hyperbolic sine).
    """

    def __init__(self, x) -> None:
        super(asinh, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise inverse hyperbolic sine of x.
        """
        return np.arcsinh(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        # Always unknown.
        raise NotImplementedError("sign_from_args not implemented for asinh.")

    def is_atom_convex(self) -> bool:
        return False

    def is_atom_concave(self) -> bool:
        return False

    def is_atom_esr(self) -> bool:
        return True

    def is_atom_hsr(self) -> bool:
        return True

    def is_incr(self, idx) -> bool:
        return True

    def is_decr(self, idx) -> bool:
        return False

    def _domain(self) -> List[Constraint]:
        return []

    def _grad(self, values) -> List[Constraint]:
        raise NotImplementedError("Gradient not implemented for asinh.")

    def _verify_hess_vec_args(self):
        return isinstance(self.args[0], Variable)

    def _hess_vec(self, vec):
        x = self.args[0]
        idxs = np.arange(x.size, dtype=int)
        vals = (-x.value / (1.0 + x.value ** 2) ** 1.5).flatten(order='F') * vec
        return {(x, x): (idxs, idxs, vals)}

    def _verify_jacobian_args(self):
        return isinstance(self.args[0], Variable)

    def _jacobian(self):
        x = self.args[0]
        idxs = np.arange(x.size, dtype=int)
        vals = 1.0 / np.sqrt(1.0 + x.value**2).flatten(order='F')
        return {x: (idxs, idxs, vals)}

class atanh(Elementwise):
    """Elementwise :math:`\\operatorname{atanh} x` (inverse hyperbolic tangent).
    """

    def __init__(self, x) -> None:
        super(atanh, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise inverse hyperbolic tangent of x.
        """
        return np.arctanh(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        # Always unknown.
        raise NotImplementedError("sign_from_args not implemented for atanh.")

    def is_atom_convex(self) -> bool:
        return False

    def is_atom_concave(self) -> bool:
        return False

    def is_atom_esr(self) -> bool:
        return True

    def is_atom_hsr(self) -> bool:
        return True

    def is_incr(self, idx) -> bool:
        return True

    def is_decr(self, idx) -> bool:
        return False

    def _domain(self) -> List[Constraint]:
        return [self.args[0] < 1, self.args[0] > -1]

    def _grad(self, values) -> List[Constraint]:
        raise NotImplementedError("Gradient not implemented for atanh.")

    def _verify_hess_vec_args(self):
        return isinstance(self.args[0], Variable)

    def _hess_vec(self, vec):
        x = self.args[0]
        idxs = np.arange(x.size, dtype=int)
        vals = (2.0 * x.value / (1.0 - x.value ** 2) ** 2).flatten(order='F') * vec
        return {(x, x): (idxs, idxs, vals)}

    def _verify_jacobian_args(self):
        return isinstance(self.args[0], Variable)

    def _jacobian(self):
        x = self.args[0]
        idxs = np.arange(x.size, dtype=int)
        vals = 1.0 / (1.0 - x.value**2).flatten(order='F')
        return {x: (idxs, idxs, vals)}