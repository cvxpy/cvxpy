"""
Copyright 2025 Youssouf Emine

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
from functools import wraps

from cvxpy.atoms.atom import Atom


class Squeeze(Atom):
    """
    Squeeze an array by removing single-dimensional entries from the shape.
    This is equivalent to np.squeeze.
    """
    _allow_complex = True

    def __init__(self, expr, axis=None) -> None:
        self.axis = axis
        super(Squeeze, self).__init__(expr)

    def shape_from_args(self) -> tuple[int, ...]:
        """
        Returns the shape of the squeezed expression.
        If axis is specified, only those dimensions are removed.
        """
        shape = list(self.args[0].shape)
        if self.axis is None:
            return tuple(s for s in shape if s > 1)
        axes = self.axis if isinstance(self.axis, list) else [self.axis]
        for ax in axes:
            if ax < 0:
                ax += len(shape)
            if ax < 0 or ax >= len(shape):
                raise ValueError(f"axis {ax} is out of bounds for array of dimension {len(shape)}")
            if shape[ax] != 1:
                raise ValueError(f"Cannot squeeze axis {ax} with size {shape[ax]}.")
            shape[ax] = 0
        return tuple(s for s in shape if s > 0)


@wraps(Squeeze)
def squeeze(expr, axis=None):
    """
    Wrapper for the Squeeze class.
    """
    return Squeeze(expr, axis)