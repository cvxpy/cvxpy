Version 1.0 (targets)
---------------------
* TODO change *args to lists.
* TODO change OPTIMAL to SOLVED?
* TODO change SDP to only set upper triag == lower triag.
* TODO separate out parts of DCPAttr and refactor so universal rules used for affine atoms.
* TODO change size to size[0]*size[1] and shape to size.
* TODO add integer/boolean variables.

Version 0.2.16 (next release)
---------------------------
* Added log1p.
* Added scalene penalty.
* Made LinOp for multiplication on the right.
* TODO break up DCPAttr so only use atom DCP function.
* Added boolean and integer variables.
* Made geo_mean elementwise.

Version 0.2.15
------------
* Made it so you can assign a value to a variable for an initial value.
* Overloaded power operator.

Version 0.2.14
------------
* Added unpack_results function to update problem state given results from a solver.
* Fixed bug where parameters were cached.


Version 0.2.13
------------
* Got rid of NumPy 1D array warning.
* Made vstack and hstack not create new variables.
* Replaced memoize with lazyprop in constraints.
* Refactored code for caching problem parsing.

Version 0.2.12
------------
* Changed cvxpy to use new SCS interface.

Version 0.2.11
------------
* Renamed semidefinite to Semidef.
* Switched solver_specific_opts to **kwargs.
* Added vec and documentation for reshape and vec.
* Changed repr to print names in line with class names.
* Changed str for Problem.
* Made < and > map to <= and >=.
* Added warning for NumPy 1D arrays.

Version 0.2.10
------------
* CVXPY throws an error when a solver error is encountered.
* Presolver removes constraints with no variables.
* Conversion from non-linear constraints to linear constraints is cached. This fixes issue #120.2.

Version 0.2.9
-----------
* Fixed bug with sign multiplication.
* Added check that objective is Minimize/Maximize.
* Fixed bug with key error when solving exponential cone problem with CVXOPT.

Version 0.2.8
------------
* Removed a stray println.

Version 0.2.7
------------------------------
* CVXPY import can succeed even if SCS import fails.
* The sign of vector and matrix constant is positive (negative) if all the entries are positive (negative), instead of always being unknown.
* Can now use negative indices.
* Added *_INACCURATE return codes.

Version 0.2.6
-----------
* Made all lin_to_matrix functions return SciPy sparse matrix or NumPy matrix (instead of ndarray).