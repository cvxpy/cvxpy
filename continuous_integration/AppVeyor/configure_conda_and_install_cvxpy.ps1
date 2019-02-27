conda config --add channels conda-forge oxfordcontrol
conda create -n testenv --yes python=$env:PYTHON_VERSION mkl=2018.0.3 pip nose numpy scipy
conda activate testenv
conda install -c conda-forge --yes lapack ecos multiprocess
conda install -c conda-forge --yes scs
# ^ There is actually no Windows version of SCS on conda-forge.
#
#   When we eventually install cvxpy from source, pip processes dependencies
#	by finding a Windows-compatible version of SCS on pypi. 
#
#   There is a version of SCS on the cvxgrp conda channel, but that version
#   doesn't link properly with MKL upon installation (which causes SCS to
#   fail on all SDP tests with matrix variables of order > 2). We wouldnt 
#   want to install that version of SCS here, since we havent run cvxpy's
#   test-suite yet.
#
#   Im leaving this statement as-is. For now it serves as a reminder that
#   proper conda Windows support likely requires a conda version of SCS
#   that comes with linked BLAS and LAPACK libraries.
#
#   We * can * still get partial conda+Windows support, by running conda-build
#	with the cvxgrp channel (so that it grabs the poorly linked SCS 1.2.6).
#   Users wont be able to run the cvxpy tests without upgrading SCS, but they
#   should be able to at least install cvxpy. Alternatively we could just remove
#	SCS as a build and run requirement from the conda recipe (at least for Windows).
#
conda install -c anaconda --yes flake8
python setup.py install