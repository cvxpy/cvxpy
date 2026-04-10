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
import threading
from collections.abc import Generator

_thread_local = threading.local()


@contextlib.contextmanager
def dpp_scope() -> Generator[None, None, None]:
    """Context manager for DPP curvature analysis

    When this scope is active, parameters are affine, not constant.
    For example, if `param` is a Parameter, then

    ```
        with dpp_scope():
            print("param is constant: ", param.is_constant())
            print("param is affine: ", param.is_affine())
    ```

    would print

        param is constant: False
        param is affine: True
    """
    prev_state = getattr(_thread_local, 'dpp_scope_active', False)
    _thread_local.dpp_scope_active = True
    yield
    _thread_local.dpp_scope_active = prev_state


def dpp_scope_active() -> bool:
    """Returns True if a `dpp_scope` is active. """
    return getattr(_thread_local, 'dpp_scope_active', False)


@contextlib.contextmanager
def quad_form_dpp_scope() -> Generator[None, None, None]:
    """Context manager for quad_form DPP analysis with QP solvers.

    When active, QuadForm.is_atom_convex/concave() relaxes the P.is_constant()
    requirement, allowing parametric P matrices for QP solvers that can handle
    quadratic objectives directly.

    This scope is entered by the solving chain when a QP solver is selected,
    enabling DPP caching for problems like min x'Px where P is a Parameter.
    """
    prev = getattr(_thread_local, 'quad_form_dpp_scope_active', False)
    _thread_local.quad_form_dpp_scope_active = True
    yield
    _thread_local.quad_form_dpp_scope_active = prev


def quad_form_dpp_scope_active() -> bool:
    """Returns True if a `quad_form_dpp_scope` is active."""
    return getattr(_thread_local, 'quad_form_dpp_scope_active', False)
