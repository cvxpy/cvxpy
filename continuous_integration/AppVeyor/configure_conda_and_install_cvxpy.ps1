conda config --add channels conda-forge
conda config --add channels oxfordcontrol
conda create -n testenv --yes python=$env:PYTHON_VERSION mkl=2018.0.3 pip nose numpy scipy
conda activate testenv
"python=$env:PYTHON_VERSION" | Out-File C:\conda\envs\testenv\conda-meta\pinned -encoding ascii
conda install --yes lapack ecos multiprocess
conda install -c conda-forge --yes scs
conda install -c anaconda --yes flake8
pip install diffcp
python setup.py install
