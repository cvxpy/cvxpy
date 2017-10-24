# cvxpy2sdpa
cvxpy interface for SDPA solver for Semidefinite programming.
Developed from SDPA-P original SDPA python interface.

This interface was tested only under linux platforms.

For installation, please install sdpa manually following the steps in http://sdpa.sourceforge.net/.

Once you have installed sdpa in sdpadir you can go to cvxpy2sdpa folder an locate setup.py arquive.

Modify the paths of SDPA_DIR to your spda directory intalled.

If you installed SDPA with a different mumps library than the sdpa provided, please modifiy all the entrities related to MUMPS_* . 

Then modify the LAPACK_DIR and BLAS_DIR, put the same lapack and blas dir paths that you use for install sdpa.

Then perform python setup.py install and use SDPA as a solver in cvxpy


