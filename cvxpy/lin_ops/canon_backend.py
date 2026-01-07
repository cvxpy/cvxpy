"""
Copyright 2022, the CVXPY authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Backwards compatibility shim for canon_backend.

All implementation has been moved to cvxpy.lin_ops.backends.
This module re-exports all public classes for backwards compatibility.
"""
from cvxpy.lin_ops.backends import (
    CanonBackend,
    Constant,
    DictTensorView,
    PythonCanonBackend,
    TensorRepresentation,
    TensorView,
)
from cvxpy.lin_ops.backends.rust_backend import RustCanonBackend
from cvxpy.lin_ops.backends.scipy_backend import (
    SciPyCanonBackend,
    SciPyTensorView,
)

__all__ = [
    "Constant",
    "TensorRepresentation",
    "CanonBackend",
    "PythonCanonBackend",
    "TensorView",
    "DictTensorView",
    "SciPyCanonBackend",
    "SciPyTensorView",
    "RustCanonBackend",
]
