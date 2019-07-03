conda config --add channels conda-forge
conda config --add channels oxfordcontrol
conda create -n testenv --yes python=$env:PYTHON_VERSION mkl=2018.0.3 pip nose numpy scipy
conda activate testenv
"python=$env:PYTHON_VERSION" | Out-File C:\conda\envs\testenv\conda-meta\pinned -encoding ascii
Get-Content -Path C:\conda\envs\testenv\conda-meta\pinned
conda install --yes lapack ecos multiprocess
pip install scs<=2.0
conda install -c anaconda --yes flake8
python setup.py install


"Some Text on first line" | Out-File C:\filename1.txt -encoding ascii
