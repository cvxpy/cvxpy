This file has not been maintained since 2018.

Version 1.0 (targets)
---------------------
* TODO change *args to lists.
* TODO change size to size[0]*size[1] and shape to size.
* TODO return 2D arrays instead of matrices.
* TODO add sets and domains for Parameters/Variables that affect DCP properties.
* TODO PSD parameters?
* TODO warmstart based on variable values.
* TODO change diag for matrices to diagm so can choose which diagonal to access.
* TODO make sure Bool.value = ? etc satisfy constraints.
* TODO make upper_tri_to_full atom?
* TODO parameterize upper_tri (so takes elems above kth diagonal).
* TODO replace norm with vecnorm and norm (for matrices) (see how numpy does it).
* TODO separate constant and linear components of expressions so potentially can cache matrix factorizations.
* TODO add informative errors for single constraint in problem instead of list, etc.
* TODO sort out norms. i.e. norm(X, 1) is wrong.
* TODO support more axis operations (norm, log_sum_exp, sum_squares).
* TODO add Lambert W function  W(z) = max x, exp(-entr(x)) <= z.
* TODO change reshape to take size tuple argument.
* TODO make identical to numpy in every behavior (ndarrays instead of matrices etc.)
* TODO make @ work when imported from future.

Version 0.4.9 (next release)
----------------------------
* TODO make division work elementwise (and with division of scalar by variable)
* TODO test and fix updating constraints.
* TODO improve error message when solving with missing parameter values.
* TODO add domains for special variables like Semidef and NonNegative (var with domain).
Canonicalize then add in domains separately.
* TODO force variable value assignments to satisfy variable properties.
* TODO make quad_form(pos cvx, pos def pos) work.
* TODO add cummax/cummin.
* TODO canonicalize to QP hack (split objective into quad + PWL + constant)

Version 0.4.8
-------------
* Fixed test with __nonzero__ called in Python 3.

Version 0.4.7
-------------
* Fixed bug with power of negative values outside domain.
* Fixed bug with __nonzero__ being removed on Python 3 by conda build.

Version 0.4.6
-------------
* Made cumsum definition implicit and O(n).
* Error for parameter P to quad_form.
* Fix for Gurobi interface and Python 3.
* Fixed bug with grad of sparse*var.
* Added .is_pwl() # Piece-wise linear.

Version 0.4.5
-------------
* Add residual to constraints (vector/matrix), use for violation.
* Added cumsum. 

Version 0.4.4
-------------
* Dropped LS as default solver.

Version 0.4.3
-------------
* Dropped dependency on CVXOPT.

Version 0.4.2 
-------------
* Fixed bug with gradient of expressions where a variable only appears in a constant term.
* Added special solver for linearly constrained least-squares problems.

Version 0.4.1 
-------------
* Made error message for chaining constraints clearer.
* Switched from toolz.memoize to fastcache.cru_cache to fix memory leak in Python 3.
* Added support for matmul and rmatmul.

Version 0.4.0
-------------
* Added domains and gradients.
* Made curvature, sign recursive and affine == convex + concave || constant. Eliminated old DCPAttr system. Memoized important info.

Version 0.3.9
-------------
* Fixed bug in diag with row vectors.
* Fixed kl_div to be elementwise.

Version 0.3.8
-------------
* Fixed bug with cvxopt solver on windows (conversion from scipy to cvxopt sparse
matrix failed).

Version 0.3.7
-------------
* Fixed bug where partial optimize didn't work with maximize objective.
* log_sum_exp axis now works for rectangular matrices.

Version 0.3.6
-------------
* Fixed bug with DCP attributes of partial optimize.

Version 0.3.5
-------------
* Made to work on Windows.

Version 0.3.4
-------------
* Added indexing with boolean ndarrays and lists of indices.

Version 0.3.3
-------------
* Converted indices and slices to ints always.
* Adding axis argument to *_entries atoms.

Version 0.3.2
-------------
* Requires cvxcore 0.0.21

Version 0.3.1
-------------
* Fixed Gurobi interface.
* Removed failing test that expected CVXOPT solver failure.
* Added logistic function.

Version 0.3.0
-------------
* Added cvxcore integration.

Version 0.2.28
--------------
* Fixed bug with partial_optimize not working for nonlinear constraints.

Version 0.2.27
--------------
* Put partial_optimize in cvxpy namespace.

Version 0.2.26
--------------
* Fixed bug with printing slices.
* Added MOSEK SOCP interface thanks to Enzo.
* Fixed bug with mul_elemwise by scalar.
* Fixed compatibility issue with power atom and new NumPy.
* Fixed Windows compatibility issue.

Version 0.2.25
--------------
* Added NonNegative variables.
* Added partial_optimize as first transform.
* Require ECOS 2.0.

Version 0.2.24
--------------
* Added Elemental interface.
* Can add problems and objectives.
* Basic remove redundant rows for CVXOPT chol. Made 'chol' default kktsolver.
* Added symmetric variables and positive definite inequalities (<<, >>).
* Added bmat atom for making block matrices.
* Added kron.

Version 0.2.23
--------------
* Made to work with SCS 1.1.3.

Version 0.2.22
--------------
* Fixed issue where using "is" instead of "==".
* Required SCS version 1.0.7 so 1.1.0 can be updated on pip.

Version 0.2.21
--------------
* Made operator overloading work with scipy sparse matrices (with scipy 0.15).
* Removed Expression shape function.
* Removed Expression __array__ function.
* Caught c.T*x where c is a NumPy 1D array.
* Added power.

Version 0.2.20
--------------
* sum no longer crashes on scalar expressions,
though you shouldn't use it.
* Added mixed integer and SOCP support to Gurobi.
* Added geo_mean

Version 0.2.19
--------------
* Fixed Python 3 runtime error where modifying the
cvxopt.solvers.options dictionary while reading it.

Version 0.2.18
--------------
* Requires ECOS 1.1.1.
* Removed norm_largest.
* Added GLPK interface.
* Updated ECOS BB exit codes.
* Factored out ability to get parameterized entries of matrix.
* Added GLPK_MIP interface.

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
