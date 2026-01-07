"""
Copyright 2025, the CVXPY authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Canonicalization backends for CVXPY.

This package provides multiple backends for canonicalizing optimization problems:
- SciPyCanonBackend: Default sparse backend using scipy sparse matrices
- CooCanonBackend: 3D COO tensor backend optimized for large DPP problems
- RustCanonBackend: Experimental Rust-accelerated backend
"""
from cvxpy.lin_ops.backends.base import (
    CanonBackend,
    Constant,
    DictTensorView,
    PythonCanonBackend,
    TensorRepresentation,
    TensorView,
)
from cvxpy.lin_ops.backends.coo_backend import (
    CooCanonBackend,
    CooTensor,
    CooTensorView,
)
from cvxpy.lin_ops.backends.rust_backend import RustCanonBackend
from cvxpy.lin_ops.backends.scipy_backend import (
    SciPyCanonBackend,
    SciPyTensorView,
)
from cvxpy.settings import (
    COO_CANON_BACKEND,
    RUST_CANON_BACKEND,
    SCIPY_CANON_BACKEND,
)

# Backend registry for get_backend()
_BACKEND_REGISTRY = {
    SCIPY_CANON_BACKEND: SciPyCanonBackend,
    RUST_CANON_BACKEND: RustCanonBackend,
    COO_CANON_BACKEND: CooCanonBackend,
}


def get_backend(backend_name: str, *args, **kwargs) -> CanonBackend:
    """
    Get a canonicalization backend by name.

    Parameters
    ----------
    backend_name : str
        Name of the backend ('SCIPY', 'COO', or 'RUST')
    *args, **kwargs
        Arguments passed to the backend constructor

    Returns
    -------
    CanonBackend
        Initialized backend instance
    """
    if backend_name not in _BACKEND_REGISTRY:
        raise KeyError(f"Unknown backend: {backend_name}")
    return _BACKEND_REGISTRY[backend_name](*args, **kwargs)




__all__ = [
    # Base classes
    "Constant",
    "TensorRepresentation",
    "CanonBackend",
    "PythonCanonBackend",
    "TensorView",
    "DictTensorView",
    # SciPy backend
    "SciPyCanonBackend",
    "SciPyTensorView",
    # COO backend
    "CooCanonBackend",
    "CooTensor",
    "CooTensorView",
    # Rust backend
    "RustCanonBackend",
    # Utility
    "get_backend",
]
