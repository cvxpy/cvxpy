Version 1.0 (targets)
---------------------
* TODO change *args to lists.
* TODO change OPTIMAL to SOLVED?
* TODO change SDP to only set upper triag == lower triag.
* TODO separate out parts of DCPAttr and refactor so universal rules used for affine atoms.
* TODO change size to size[0]*size[1] and shape to size.
* TODO add integer/boolean variables.
* TODO return 2D arrays instead of matrices.
* TODO add sets and domains for Parameters/Variables that affect DCP properties.
* TODO PSD parameters?
* TODO warmstart based on variable values.
* TODO change diag for matrices to diagm so can choose which diagonal to access.
* TODO make sure Bool.value = ? etc satisfy constraints.
* TODO make upper_tri_to_full atom?
* TODO parameterize upper_tri (so takes elems above kth diagonal).
* TODO replace norm with vecnorm and norm (for matrices) (see how numpy does it).
* TODO redo tree_mat so it uses SMs (and doesn't compress the left hand side of A*x into a matrix).
* TODO separate constant and linear components of expressions so potentially can cache matrix factorizations.
* TODO add logistic_loss

Version 0.2.18 (next release)
-----------------------------
* Requires ECOS 1.1.1.
* Removed norm_largest.


Version 0.2.17
--------------
* Optimizes rmul to mul.
* Added warmstart.
* Disabled MIP tests until nondeterminism is resolved.
* Added sum_largest, sum_smallest, norm_largest, lambda_sum_largest, lambda_sum_largest, and lambda_sum_smallest.

Version 0.2.16
--------------
* Added log1p.
* Added scalene penalty.
* Made LinOp for multiplication on the right.
* TODO break up DCPAttr so only use atom DCP function.
* Added boolean and integer variables.
* Made geo_mean elementwise.
* Python 3 support!!!
* Error checking for x**p.
* Added upper_tri.
* Changed BoolVar, IntVar to Bool, Int.
* Simplified Huber loss graph_implementation.

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