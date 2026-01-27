"""
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
    frame = sys._getframe(1)  # caller of warn()
    level = 2  # stacklevel=2 already points at the caller of warn()
    while frame is not None:
        if not _is_internal_frame(frame.f_code.co_filename):
            break
        level += 1
        frame = frame.f_back
    warnings.warn(message, category, stacklevel=level)
