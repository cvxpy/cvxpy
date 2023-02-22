"""
Copyright, the CVXPY authors

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
import contextlib
from typing import Generator

_dpp_scope_active = False


@contextlib.contextmanager
def dpp_scope() -> Generator[None, None, None]:
    """Context manager for DPP curvature analysis

    When this scope is active, parameters are affine, not constant. The
    argument For example, if `param` is a Parameter, then

    ```
        with dpp_scope():
            print("param is constant: ", param.is_constant())
            print("param is affine: ", param.is_affine())
    ```

    would print

        param is constant: False
        param is affine: True
    """
    global _dpp_scope_active
    prev_state = _dpp_scope_active
    _dpp_scope_active = True
    yield
    _dpp_scope_active = prev_state


def dpp_scope_active() -> bool:
    """Returns True if a `dpp_scope` is active. """
    return _dpp_scope_active
