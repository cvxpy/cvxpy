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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple

if TYPE_CHECKING:
    from cvxpy.expressions.constants.parameter import Parameter

import scipy.sparse as sp

import cvxpy.lin_ops.lin_utils as lu
from cvxpy import settings as s
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.leaf import Leaf
from cvxpy.utilities import scopes


class Variable(Leaf):
    """The optimization variables in a problem.

    Attributes
    ----------
    sample_bounds : tuple[np.ndarray, np.ndarray] | None
        Explicit bounds ``(low, high)`` for random initial point sampling in
        ``best_of`` NLP solves.  When set, overrides the variable's ``value``
        during random initialization.  When ``None`` and finite ``bounds`` are
        present, those are used instead.
    """

    def __init__(
        self, shape: int | Iterable[int] = (), name: str | None = None,
        var_id: int | None = None, **kwargs: Any
    ):
        if var_id is None:
            self.id = lu.get_id()
        else:
            self.id = var_id
        if name is None:
            self._name = "%s%d" % (s.VAR_PREFIX, self.id)
        elif isinstance(name, str):
            self._name = name
        else:
            raise TypeError("Variable name %s must be a string." % name)

        self._value = None
        self.delta = None
        self.gradient = None
        self.sample_bounds = None
        super(Variable, self).__init__(shape, **kwargs)

    def name(self) -> str:
        """The name of the variable."""
        return self._name

    def is_constant(self) -> bool:
        return False

    @property
    def grad(self) -> Optional[dict[Variable, sp.csc_array]]:
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.
        """
        # TODO(akshayka): Do not assume shape is 2D.
        return {self: sp.eye_array(self.size, format='csc')}

    def variables(self) -> list[Variable]:
        """Returns itself as a variable."""
        return [self]

    def parameters(self) -> list[Parameter]:
        """Returns parameters present in expression bounds, if any."""
        params = []
        if self.attributes.get('bounds') is not None:
            for b in self.attributes['bounds']:
                if isinstance(b, Expression):
                    params.extend(b.parameters())
        return params

    def is_dcp(self, dpp: bool = False) -> bool:
        """Check DCP compliance, including parameter-affine bounds."""
        if dpp and self.attributes.get('bounds') is not None:
            with scopes.dpp_scope():
                for b in self.attributes['bounds']:
                    if isinstance(b, Expression) and not b.is_affine():
                        return False
        return True

    def is_dgp(self, dpp: bool = False) -> bool:
        """Check DGP compliance, including log-log-affine bounds."""
        if dpp and self.attributes.get('bounds') is not None:
            with scopes.dpp_scope():
                for b in self.attributes['bounds']:
                    if isinstance(b, Expression) and not b.is_log_log_affine():
                        return False
        # Use base class logic: check log-log convexity/concavity
        return self.is_log_log_convex() or self.is_log_log_concave()

    def is_dpp(self, context: str = 'dcp') -> bool:
        """Check that the variable is DPP in the given context."""
        if context == 'dcp':
            return self.is_dcp(dpp=True)
        elif context == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError(f'Unsupported context {context}')

    def canonicalize(self) -> Tuple[Expression, list[Constraint]]:
        """Returns the graph implementation of the object."""
        obj = lu.create_var(self.shape, self.id)
        return (obj, [])

    def set_variable_of_provenance(self, variable: Variable) -> None:
        """Deprecated: use set_leaf_of_provenance instead."""
        self.set_leaf_of_provenance(variable)

    def variable_of_provenance(self) -> Optional[Variable]:
        """Deprecated: use leaf_of_provenance instead."""
        return self.leaf_of_provenance()

    def __repr__(self) -> str:
        """String to recreate the variable."""
        attr_str = self._get_attr_str()
        return f"Variable({self.shape}, {self.__str__()}{attr_str})"
