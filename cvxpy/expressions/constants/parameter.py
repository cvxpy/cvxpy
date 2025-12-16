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

import cvxpy.lin_ops.lin_utils as lu
from cvxpy import settings as s
from cvxpy.expressions.leaf import Leaf
from cvxpy.utilities import scopes


def is_param_affine(expr) -> bool:
    """Returns true if expression is parameters-affine (and variable-free)"""
    with scopes.dpp_scope():
        return not expr.variables() and expr.is_affine()


def is_param_free(expr) -> bool:
    """Returns true if expression is not parametrized."""
    return not expr.parameters()


class Parameter(Leaf):
    """Parameters in optimization problems.

    Parameters are constant expressions whose value may be specified
    after problem creation. The only way to modify a problem after its
    creation is through parameters. For example, you might choose to declare
    the hyper-parameters of a machine learning model to be Parameter objects;
    more generally, Parameters are useful for computing trade-off curves.

    Batch Mode
    ----------
    Parameters support batch mode for solving multiple problem instances
    efficiently. Use ``set_value(arr, batch=True)`` to set batched values
    where the leading dimensions are batch dimensions::

        param = cp.Parameter((m, n))
        param.set_value(np.random.randn(batch_size, m, n), batch=True)
        prob.solve()  # Solves batch_size problems

    The batch shape is inferred from the value shape minus the parameter shape.
    """
    PARAM_COUNT = 0

    def __init__(
        self, shape: int | tuple[int, ...] = (), name: str | None = None, value=None,
        id=None, **kwargs
    ) -> None:
        if id is None:
            self.id = lu.get_id()
        else:
            self.id = id
        if name is None:
            self._name = f"{s.PARAM_PREFIX}{self.id}"
        else:
            self._name = name
        # Initialize with value if provided.
        self._value = None
        self.delta = None
        self.gradient = None
        super(Parameter, self).__init__(shape, value, **kwargs)
        self._is_constant = True

    def set_value(self, val, batch: bool = False) -> None:
        """Set parameter value, optionally with batch dimensions.

        Parameters
        ----------
        val : array-like
            The value to assign. If batch=True, leading dimensions are
            treated as batch dimensions.
        batch : bool, optional
            If True, treat leading dimensions as batch dimensions.
            The batch_shape is inferred as val.shape[:-self.ndim]
            (or val.shape if self.ndim == 0).
        """
        import cvxpy.interface as intf

        if val is None:
            self._value = None
            self._batch_shape = ()
            return

        val = intf.convert(val)

        if batch:
            # Infer batch shape from the difference in dimensions
            if self.ndim > 0:
                if val.ndim < self.ndim:
                    raise ValueError(
                        f"Value has {val.ndim} dimensions but parameter has "
                        f"{self.ndim} dimensions. Cannot infer batch shape."
                    )
                self._batch_shape = val.shape[:-self.ndim]
                problem_shape = val.shape[-self.ndim:]
            else:
                # Scalar parameter - all dimensions are batch
                self._batch_shape = val.shape
                problem_shape = ()

            # Validate that problem dimensions match
            if problem_shape != self.shape:
                raise ValueError(
                    f"Value has problem shape {problem_shape} but parameter "
                    f"has shape {self.shape}."
                )
            self._value = val
        else:
            # Non-batched mode - use standard validation
            self._batch_shape = ()
            self.value = val  # Use the property setter for validation

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.shape, self._name, self.value, self.id, self.attributes]

    def name(self) -> str:
        return self._name

    def is_constant(self) -> bool:
        if scopes.dpp_scope_active():
            return False
        return True

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        return {}

    def parameters(self) -> list[Parameter]:
        """Returns itself as a parameter.
        """
        return [self]

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_param(self.shape, self.id)
        return (obj, [])

    def __repr__(self) -> str:
        """String to recreate the object.
        """
        attr_str = self._get_attr_str()
        if len(attr_str) > 0:
            return "Parameter(%s%s)" % (self.shape, attr_str)
        else:
            return "Parameter(%s)" % (self.shape,)
