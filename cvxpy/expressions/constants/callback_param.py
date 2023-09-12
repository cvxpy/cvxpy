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

from typing import Callable

from cvxpy.expressions.constants.parameter import Parameter


class CallbackParam(Parameter):
    """
    A parameter whose value is derived from a callback function.

    Enables writing replacing expression that would not be DPP
    by a new parameter that automatically updates its value.

    Example:
    With p and q parameters, p * q is not DPP, but
    pq = CallbackParameter(callback=lambda: p.value * q.value) is DPP.

    This is useful when only p and q should be exposed
    to the user, but pq is needed internally.
    """

    def __init__(self, callback: Callable, shape: int | tuple[int, ...] = (), **kwargs) -> None:
        """
        callback: function that returns the value of the parameter.
        """

        self._callback = callback
        super(CallbackParam, self).__init__(shape, **kwargs)

    @property
    def value(self):
        """Evaluate the callback to get the value.
        """
        return self._validate_value(self._callback())

    @value.setter
    def value(self, _val):
        raise NotImplementedError("Cannot set the value of a CallbackParam.")
