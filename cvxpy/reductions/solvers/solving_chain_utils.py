import warnings

from cvxpy.settings import CPP_CANON_BACKEND, NUMPY_CANON_BACKEND, SCIPY_CANON_BACKEND


def get_canon_backend(problem, canon_backend: str) -> str:
    """
    This function checks if the problem has expressions of dimension greater
    than 2 or if it lacks C++ support, then raises a warning if the default
    backend is not specified or raises an error if the backend is specified
    as 'CPP'.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    canon_backend : str
        'CPP' (default) | 'SCIPY'
        Specifies which backend to use for canonicalization, which can affect
        compilation time. Defaults to None, i.e., selecting the default
        backend.
    Returns
    -------
    canon_backend : str
        The canonicalization backend to use.
    """

    if not problem._supports_cpp():
        if canon_backend is None:
            warnings.warn(UserWarning(
                f"The problem includes expressions that don't support {CPP_CANON_BACKEND} backend. "
                f"Defaulting to the {SCIPY_CANON_BACKEND} backend for canonicalization."))
            return SCIPY_CANON_BACKEND
        if canon_backend == CPP_CANON_BACKEND:
            raise ValueError(f"The {CPP_CANON_BACKEND} backend cannot be used with problems "
                             f"that have expressions which do not support it.")
        return canon_backend  # Use the specified backend (e.g., SCIPY_CANON_BACKEND)

    if problem._max_ndim() > 2:
        if canon_backend is None:
            warnings.warn(UserWarning(
                f"The problem has an expression with dimension greater than 2. "
                f"Defaulting to the {SCIPY_CANON_BACKEND} backend for canonicalization."))
            return SCIPY_CANON_BACKEND
        if canon_backend == CPP_CANON_BACKEND:
            raise ValueError(f"Only the {SCIPY_CANON_BACKEND} and {NUMPY_CANON_BACKEND} "
                             f"backends are supported for problems with expressions of "
                             f"dimension greater than 2.")
    return canon_backend