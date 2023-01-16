"""
Copyright 2017 Steven Diamond

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

import cvxpy.utilities.performance_utils as perf
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression


class indicator(Expression):
    """An expression representing the convex function I(constraints) = 0
       if constraints hold, +infty otherwise.

    Parameters
    ----------
    constraints : list
       A list of constraint objects.
    err_tol:
       A numeric tolerance for determining whether the constraints hold.
    """

    def __init__(self, constraints: List[Constraint], err_tol: float = 1e-3) -> None:
        self.args = constraints
        self.err_tol = err_tol
        super(indicator, self).__init__()

    @perf.compute_once
    def is_constant(self) -> bool:
        """The Indicator is constant if all constraints have constant args.
        """
        all_args = sum([c.args for c in self.args], [])
        return all([arg.is_constant() for arg in all_args])

    def is_convex(self) -> bool:
        """Is the expression convex?
        """
        return True

    def is_concave(self) -> bool:
        """Is the expression concave?
        """
        return False

    def is_log_log_convex(self) -> bool:
        return False

    def is_log_log_concave(self) -> bool:
        return False

    def is_nonneg(self) -> bool:
        """Is the expression positive?
        """
        return True

    def is_nonpos(self) -> bool:
        """Is the expression negative?
        """
        return False

    def is_imag(self) -> bool:
        """Is the Leaf imaginary?
        """
        return False

    def is_complex(self) -> bool:
        """Is the Leaf complex valued?
        """
        return False

    def get_data(self) -> List[float]:
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.err_tol]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the (row, col) dimensions of the expression.
        """
        return ()

    def name(self) -> str:
        """Returns the string representation of the expression.
        """
        return f"Indicator({self.args})"

    def domain(self) -> List[Constraint]:
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        return self.args

    @property
    def value(self) -> float:
        """Returns the numeric value of the expression.

        Returns:
            A numpy matrix or a scalar.
        """
        if all(cons.value(tolerance=self.err_tol) for cons in self.args):
            return 0.0
        else:
            return np.infty

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.
        None indicates variable values unknown or outside domain.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        # TODO
        raise NotImplementedError()
