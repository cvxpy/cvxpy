conda config --add channels conda-forge
conda config --add channels oxfordcontrol
conda create -n testenv --yes python=$env:PYTHON_VERSION mkl=2018.0.3 pip nose numpy scipy
conda activate testenv
conda install --yes lapack ecos multiprocess
conda install --yes scs
# ^ There is actually no Windows version of SCS on conda-forge, or oxfordcontrol.
#	So in principle, the last install command above wont do anything. I'm
#   leaving it as is, because it serves as a reminder of the following facts:
#
#   (1) There is a version of SCS on the cvxgrp conda channel, but that version
#   doesn't link properly with MKL upon installation (which causes SCS to
#   fail on all SDP tests with matrix variables of order > 2). We wouldnt 
#   want to install that version of SCS here, since we havent run cvxpy's
#   test-suite yet.
#
#   (2) When we eventually install cvxpy from source, pip processes dependencies
#	by finding a Windows-compatible version of SCS on pypi. When it does this,
#	it ends up installing a properly linked instance of SCS version >= 2.0. 
#
#   When we need to create the cvxpy conda package (for uploading to the cvxgrp
#   conda channel) we will settle for the poorly linked version of SCS in
#   point (1) above. Users wont be able to run the cvxpy tests without upgrading
#   SCS, but they will be able to install cvxpy, and use scs for LP/SOCP/EXP cone
#   programs.
#
conda install -c anaconda --yes flake8
python setup.py install