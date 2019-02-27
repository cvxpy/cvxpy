conda create -n testenv --yes python=$env:PYTHON_VERSION mkl=2018.0.3 pip nose numpy scipy
conda activate testenv
conda config --add channels conda-forge
conda install -c conda-forge --yes lapack
conda install -c conda-forge --yes ecos scs multiprocess
conda install -c anaconda --yes flake8
python setup.py install