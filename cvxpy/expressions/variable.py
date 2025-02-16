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

from typing import Any, Iterable, Optional, Tuple

import scipy.sparse as sp

import cvxpy.lin_ops.lin_utils as lu
from cvxpy import settings as s
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.leaf import Leaf


class Variable(Leaf):
    """The optimization variables in a problem."""

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

        self._variable_with_attributes: Variable | None = None
        self._value = None
        self.delta = None
        self.gradient = None
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

    def canonicalize(self) -> Tuple[Expression, list[Constraint]]:
        """Returns the graph implementation of the object."""
        obj = lu.create_var(self.shape, self.id)
        return (obj, [])

    def attributes_were_lowered(self) -> bool:
        """True iff variable generated when lowering a variable with attributes."""
        return self._variable_with_attributes is not None

    def set_variable_of_provenance(self, variable: Variable) -> None:
        assert variable.attributes
        self._variable_with_attributes = variable

    def variable_of_provenance(self) -> Optional[Variable]:
        """Returns a variable with attributes from which this variable was generated."""
        return self._variable_with_attributes

    def __repr__(self) -> str:
        """String to recreate the variable."""
        attr_str = self._get_attr_str()
        return f"Variable({self.shape}, {self.__str__()}{attr_str})"
