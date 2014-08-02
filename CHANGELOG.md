Version 2.7 (not yet released)
------------------------------
* CVXPY import can succeed even if SCS import fails.
* The sign of vector and matrix constant is positive (negative) if all the entries are positive (negative), instead of always being unknown.
* Can now use negative indices.

Version 2.6
-----------
* Made all lin_to_matrix functions return SciPy sparse matrix or NumPy matrix (instead of ndarray).