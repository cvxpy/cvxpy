Version 2.11 (next release)
---------------------------
* Renamed semidefinite to Semidefinite.
* Switched solver_specific_opts to **kwargs.
* Added vec and documentation for reshape and vec.

Version 2.10
------------
* CVXPY throws an error when a solver error is encountered.
* Presolver removes constraints with no variables.
* Conversion from non-linear constraints to linear constraints is cached. This fixes issue #122.

Version 2.9
-----------
* Fixed bug with sign multiplication.
* Added check that objective is Minimize/Maximize.
* Fixed bug with key error when solving exponential cone problem with CVXOPT.

Version 2.8
------------
* Removed a stray println.

Version 2.7
------------------------------
* CVXPY import can succeed even if SCS import fails.
* The sign of vector and matrix constant is positive (negative) if all the entries are positive (negative), instead of always being unknown.
* Can now use negative indices.
* Added *_INACCURATE return codes.

Version 2.6
-----------
* Made all lin_to_matrix functions return SciPy sparse matrix or NumPy matrix (instead of ndarray).