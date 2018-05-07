# Download miniconda

$MINICONDA_URL = "https://repo.continuum.io/miniconda/"
$fileurl = $MINICONDA_URL + $env:MINICONDA_FILENAME
$filepath = $pwd.Path + "\" + $env:MINICONDA_FILENAME
$client = new-object System.Net.WebClient
$client.DownloadFile($fileurl,  $filepath)

# Install miniconda

$install_args = "/InstallationType=AllUsers /S /RegisterPython=1 /D=" + $env:PYTHON
Write-Host $filepath $install_args
Start-Process -Filepath $filepath -ArgumentList $install_args -Wait -Passthru
# The conda install doesn't work well when called from PowerShell.
# We need to set some environment variables. First, we set them
# for the current session of PowerShell:
$dir_to_add = "${env:PYTHON};${env:PYTHON}\Scripts;"
$env:PATH = $dir_to_add + $env:PATH
# but we also need to set them globally:
setx PATH $env:PATH /m
echo $env:PATH

# Configure conda

conda create -n testenv --yes python=$env:PYTHON_VERSION mkl pip nose numpy scipy
activate testenv
# The conda activation doesn't work well from PowerShell.
# We need to update environment variables, but this need only
# be done for the current session.
$env:PATH = "${env:PYTHON}\envs\testenv;${env:PYTHON}\envs\testenv\Scripts;${env:PYTHON}\envs\testenv\Library\bin;" + $env:PATH
# The above line updates PATH for the same reason as when we installed Miniconda.
conda install -c conda-forge --yes lapack
conda install -c cvxgrp --yes ecos scs multiprocess
conda install -c anaconda --yes flake8

# Install cvxpy

python setup.py install


