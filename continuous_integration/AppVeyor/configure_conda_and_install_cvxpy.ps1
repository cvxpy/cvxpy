conda config --add channels conda-forge cvxgrp oxfordcontrol
conda create -n testenv --yes python=$env:PYTHON_VERSION mkl=2018.0.3 pip nose numpy scipy
conda activate testenv
conda install -c conda-forge --yes lapack ecos multiprocess
conda install -c cvxgrp --yes scs
conda install -c anaconda --yes flake8
python setup.py install