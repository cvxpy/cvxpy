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

Utility for issuing warnings that appear to originate from user code.

``warnings.warn`` requires a fixed ``stacklevel`` to determine which
call-site the warning is reported at. Because CVXPY's internal call
depth varies (e.g. recursive ``canonicalize_tree``), a fixed value
cannot reliably point to user code. This module provides a ``warn``
function that walks the call stack at runtime and picks the first
frame outside the cvxpy package.
"""

import os
import sys
import warnings

# Root of the cvxpy source tree (the ``cvxpy/`` package directory).
_CVXPY_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_internal_frame(filename: str) -> bool:
    """Return True if *filename* belongs to cvxpy internals (not tests)."""
    if not filename.startswith(_CVXPY_SRC):
        return False
    rel = os.path.relpath(filename, _CVXPY_SRC)
    return not rel.startswith("tests")


def warn(message, category=UserWarning):
    """Issue a warning that appears to originate from user code.

    Walks up the call stack from the caller until a frame outside the
    cvxpy package (or inside ``cvxpy/tests/``) is found, then sets
    ``stacklevel`` so the warning is attributed to that frame.
    """
    level = 2  # stacklevel=2 already points at the caller of warn()
    # sys._getframe is a CPython implementation detail; fall back to a
    # fixed stacklevel if it is unavailable.
    if not hasattr(sys, "_getframe"):
        warnings.warn(message, category, stacklevel=level)
        return
    frame = sys._getframe(1)  # caller of warn()
    # If the entire stack is internal (f_back eventually returns None),
    # level may exceed the actual stack depth.  That is fine:
    # warnings.warn gracefully handles an out-of-range stacklevel by
    # attributing the warning to "<sys>:0" rather than raising an error.
    while frame is not None:
        if not _is_internal_frame(frame.f_code.co_filename):
            break
        level += 1
        frame = frame.f_back
    warnings.warn(message, category, stacklevel=level)
