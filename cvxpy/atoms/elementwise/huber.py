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

from typing import Tuple

import numpy as np
import scipy.special

from cvxpy.atoms.atom import Atom
from cvxpy.atoms.elementwise.elementwise import Elementwise

# TODO(akshayka): DGP support.

def huber(x, M=1, t=None):
    """The Huber function with optional concomitant scale parameter.

    With two arguments, returns the standard Huber penalty::

        huber(x, M) = 2M|x| - M^2   for |x| >= M
                      |x|^2          for |x| <= M

    With three arguments, returns the perspective form for concomitant
    scale estimation (jointly convex in x and t)::

        huber(x, M, t) = t * huber(x/t, M)   for t > 0
                         +Inf                  for t <= 0

    which equals ``2M|x| - M^2 * t`` for ``|x| >= M*t`` and
    ``|x|^2 / t`` for ``|x| <= M*t``.

    Parameters
    ----------
    x : Expression
        The expression to which the huber function will be applied.
        Must be affine (the function is nonmonotonic in x).
    M : int, float, Constant, or Parameter
        Positive scalar halfwidth. Defaults to 1.
    t : Expression, Variable, Constant, or None
        Optional positive scale parameter. When provided as a Variable,
        enables concomitant scale estimation. Must be concave (or affine).
        When None or omitted, the standard two-argument huber is returned.
    """
    if t is None:
        return _HuberAtom(x, M)
    else:
        return _HuberPerspectiveAtom(x, t, M)

class _HuberAtom(Elementwise):
    """The standard two-argument Huber penalty atom.

    .. math::

        \\operatorname{Huber}(x, M) =
            \\begin{cases}
                2M|x|-M^2 & \\text{for } |x| \\geq |M| \\\\
                      |x|^2 & \\text{for } |x| \\leq |M|.
            \\end{cases}

    :math:`M` defaults to 1.

    Users should call :func:`huber` rather than instantiating this class
    directly.

    Parameters
    ----------
    x : Expression
        The expression to which the huber function will be applied.
    M : Constant or Parameter
        A non-negative scalar constant or Parameter.
    """

    def __init__(self, x, M: int = 1) -> None:
        self.M = self.cast_to_const(M)
        super(_HuberAtom, self).__init__(x)

    def parameters(self):
        """If M is a Parameter, include it in the list of Parameters."""
        return super().parameters() + self.M.parameters()

    @Elementwise.numpy_numeric
    def numeric(self, values) -> float:
        """Returns the huber function applied elementwise to x."""
        return 2 * scipy.special.huber(self.M.value, values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression."""
        # Always non-negative.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?"""
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?"""
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?"""
        return self.args[idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?"""
        return self.args[idx].is_nonpos()

    def is_quadratic(self) -> bool:
        """Quadratic if x is affine."""
        return self.args[0].is_affine()

    def has_quadratic_term(self) -> bool:
        """Always generates a quadratic term."""
        return True

    def get_data(self):
        """Returns the parameter M."""
        return [self.M]

    def validate_arguments(self) -> None:
        """Checks that M >= 0 and is a constant scalar."""
        if not (self.M.is_nonneg() and self.M.is_scalar() and self.M.is_constant()):
            raise ValueError("M must be a non-negative scalar constant or Parameter.")
        super(_HuberAtom, self).validate_arguments()

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        rows = self.args[0].size
        cols = self.size
        min_val = np.minimum(np.abs(values[0]), self.M.value)
        grad_vals = 2 * np.multiply(np.sign(values[0]), min_val)
        return [_HuberAtom.elemwise_grad_to_diag(grad_vals, rows, cols)]


class _HuberPerspectiveAtom(Atom):
    """The three-argument perspective Huber atom: ``t * huber(x/t, M)``.

    Jointly convex in ``(x, t)`` with ``t > 0``. This is the perspective
    transform of the standard Huber function and is useful for concomitant
    scale estimation.

    Users should call :func:`huber` rather than instantiating this class
    directly.

    The canonicalization uses the homogenized SOC form. For each element i:

        t * huber(x_i / t, M)  =  minimize   2*M*u_i - M^2 * t
        subject to:
            ||(2*v_i, u_i - M*t)||_2  <=  u_i + M*t
            v_i  >=  |x_i|
            t  >=  0

    which is the perspective of the standard Huber SOC representation.

    Parameters
    ----------
    x : Expression
        Affine expression (the function is nonmonotonic in x).
    M : int, float, Constant, or Parameter
        Non-negative scalar halfwidth constant.
    t : Expression
        Concave (or affine) positive scalar expression. May be a Variable
        to enable concomitant scale estimation.
    """

    def __init__(self, x, t, M=1) -> None:
        self.M = self.cast_to_const(M)
        t = self.cast_to_const(t)
        super(_HuberPerspectiveAtom, self).__init__(x, t)

    @property
    def _x(self):
        return self.args[0]

    @property
    def _t(self):
        return self.args[1]

    def get_data(self):
        """Returns the parameter M."""
        return [self.M]

    def parameters(self):
        """If M is a Parameter, include it in the list of Parameters."""
        return super().parameters() + self.M.parameters()

    def numeric(self, values):
        """Numerically evaluate t * huber(x/t, M) elementwise."""
        x_val = values[0]
        t_val = values[1]
        if np.any(t_val <= 0):
            return np.full(np.broadcast(x_val, t_val).shape, np.inf)
        return t_val * 2 * scipy.special.huber(self.M.value, x_val / t_val)

    def shape_from_args(self) -> Tuple[int, ...]:
        """The output shape is broadcast(x, t)."""
        return np.broadcast_shapes(self._x.shape, self._t.shape)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression."""
        # t * huber(x/t, M) >= 0 whenever t > 0.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Jointly convex in (x, t)."""
        return True

    def is_atom_concave(self) -> bool:
        return False

    def is_incr(self, idx) -> bool:
        """Monotonicity:
          - in x (idx=0): nonmonotonic in general (same as 2-arg huber)
          - in t (idx=1): non-increasing (perspective of a nonneg convex fn
                          is non-increasing in t when minimizing)
        """
        if idx == 0:
            # nondecreasing in x only if x is nonneg (same logic as 2-arg)
            return self._x.is_nonneg()
        else:
            # non-increasing in t
            return False

    def is_decr(self, idx) -> bool:
        if idx == 0:
            return self._x.is_nonpos()
        else:
            # non-increasing in t
            return True

    def is_quadratic(self) -> bool:
        """Not quadratic in general when t is a variable."""
        return False

    def has_quadratic_term(self) -> bool:
        return False

    def validate_arguments(self) -> None:
        """Check M is a non-negative scalar constant; t must be concave/affine."""
        if not (self.M.is_nonneg() and self.M.is_scalar() and self.M.is_constant()):
            raise ValueError("M must be a non-negative scalar constant or Parameter.")
        if not self._t.is_scalar():
            raise ValueError(
                "t must be a scalar expression. "
                "For vector x, a single scalar t scales the entire Huber loss."
            )
        if not (self._t.is_affine() or self._t.is_concave()):
            raise ValueError(
                "t must be a concave or affine expression (DCP requirement: "
                "the Huber perspective is non-increasing in t)."
            )
        super(_HuberPerspectiveAtom, self).validate_arguments()

    def _grad(self, values):
        """Gradient w.r.t. x and t.

        d/dx [t * huber(x/t, M)] = huber'(x/t, M)  (clipped linear in x/t)
        d/dt [t * huber(x/t, M)] = huber(x/t, M) - (x/t) * huber'(x/t, M)
                                  = -min(x/t, M)^2   (always <= 0)
        """
        x_val = np.asarray(values[0], dtype=float)
        t_val = float(values[1])
        if t_val <= 0:
            return [None, None]

        r = x_val / t_val
        M_val = float(self.M.value)
        clipped = np.sign(r) * np.minimum(np.abs(r), M_val)

        # gradient w.r.t. x: 2 * clipped (factor of 2 from CVXPY's huber convention)
        grad_x = 2 * clipped  # shape matches x

        # gradient w.r.t. t: sum over elements of -2 * min(|r|, M)^2
        # = sum_i [huber(x_i/t, M) - (x_i/t)*huber'(x_i/t, M)]
        # = -sum_i min(x_i/t, M)^2   [can be shown by case analysis]
        grad_t = np.sum(-2 * np.minimum(np.abs(r), M_val) ** 2)  # scalar

        return [grad_x, np.atleast_1d(np.array(grad_t))]