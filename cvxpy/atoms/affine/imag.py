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

from cvxpy.atoms.affine.affine_atom import AffAtom
import numpy as np


class imag(AffAtom):
    """Extracts the imaginary part of an expression.
    """
    def __init__(self, expr) -> None:
        super(imag, self).__init__(expr)

    def numeric(self, values):
        """Convert the vector constant into a diagonal matrix.
        """
        return np.imag(values[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the expression.
        """
        return self.args[0].shape

    def is_imag(self) -> bool:
        """Is the expression imaginary?
        """
        return False

    def is_complex(self) -> bool:
        """Is the expression complex valued?
        """
        return False

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        return self.args[0].is_hermitian()
