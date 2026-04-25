"""
Copyright 2025 CVXPY developers

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


class SolverInfo:
    """A context class that propagates solver attributes
    through the solving chain.
    """
    def __init__(self, solver=None, supported_constraints=None, supports_bounds=False,
                 psd_triangle_kind=None, psd_sqrt2_scaling=None,
                 x_cone_kinds=None):
        self.solver_name = solver
        self.solver_supported_constraints = supported_constraints
        self.solver_supports_bounds = supports_bounds
        self.psd_triangle_kind = psd_triangle_kind
        self.psd_sqrt2_scaling = psd_sqrt2_scaling
        # Cone kinds the solver consumes natively as direct-x cones
        # (XConeSpec entries on the primal variable rather than
        # cones-on-slacks).  Empty set means no native x_cone path,
        # so ExtractIdentityCones is a no-op.
        self.x_cone_kinds = frozenset(x_cone_kinds or ())
