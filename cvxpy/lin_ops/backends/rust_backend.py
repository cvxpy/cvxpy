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
"""
from __future__ import annotations

import scipy.sparse as sp

from cvxpy.lin_ops import LinOp
from cvxpy.lin_ops.backends.base import CanonBackend


class RustCanonBackend(CanonBackend):
    """
    Rust-accelerated canonicalization backend (experimental).

    TODO: This backend is a work-in-progress. The main blocker is more complete
    benchmarking and performance improvements before it can be enabled by default.

    For implementation details and progress, see:
    https://github.com/cvxpy/cvxpy/pull/3018
    """

    def build_matrix(
        self, lin_ops: list[LinOp], order: str = 'F'
    ) -> sp.csc_array:
        import cvxpy_rust
        self.id_to_col[-1] = self.var_length
        (data, (row, col), shape) = cvxpy_rust.build_matrix(lin_ops,
                                                            self.param_size_plus_one,
                                                            self.id_to_col,
                                                            self.param_to_size,
                                                            self.param_to_col,
                                                            self.var_length,
                                                            order)
        self.id_to_col.pop(-1)
        return sp.csc_array((data, (row, col)), shape)
