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

import sys

from cvxpy.utilities.warn import warn

# Packages whose macOS wheels bundle their own OpenMP runtime. Loading two of
# these into one process can crash inside a solver's parallel section:
#   - knitro and cyipopt each bundle LLVM ``libomp.dylib``
#   - cvxopt bundles a parallel OpenBLAS that links GNU ``libgomp.1.dylib``
# LLVM libomp and GNU libgomp were never designed to coexist in one process,
# and even two copies of libomp share no global state across instances.
#
# The conflict is macOS-specific: on Linux, auditwheel-built wheels typically
# share a single system libgomp via symbol versioning, and we have not observed
# a crash from this combination in CI.
_OMP_BUNDLING_PACKAGES: tuple[str, ...] = ("knitro", "cyipopt", "cvxopt")


def warn_if_omp_conflict(importing: str) -> None:
    """Warn if importing ``importing`` would put two OMP-bundling solvers in
    one process on macOS.

    Call this immediately before importing the solver's native bindings, so the
    user sees the warning before any subsequent crash. The check looks at
    ``sys.modules`` for already-imported sibling packages and does not import
    anything itself.

    No-op on non-macOS platforms, where the bundled-OpenMP conflict does not
    manifest as a crash in practice.

    Parameters
    ----------
    importing : str
        Top-level package name about to be imported, e.g. ``"knitro"``.
    """
    if not sys.platform.startswith("darwin"):
        return
    if importing not in _OMP_BUNDLING_PACKAGES:
        return
    already_loaded = [
        pkg for pkg in _OMP_BUNDLING_PACKAGES
        if pkg != importing and pkg in sys.modules
    ]
    if not already_loaded:
        return
    warn(
        f"Loading {importing!r} into a process that has already imported "
        f"{', '.join(repr(p) for p in already_loaded)}. Each of these "
        "packages ships its own OpenMP runtime in its macOS wheel (LLVM "
        "libomp for knitro and cyipopt; GNU libgomp for cvxopt via its "
        "OpenBLAS dependency), and mixing them in one process can crash "
        "inside a solver's parallel section. If you hit a segfault, run "
        "each of these solvers in a separate Python process.",
        RuntimeWarning,
    )
