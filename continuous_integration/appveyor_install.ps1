# Download miniconda

$MINICONDA_URL = "https://repo.continuum.io/miniconda/"
$fileurl = $MINICONDA_URL + $env:MINICONDA_FILENAME
$filepath = $pwd.Path + "\" + $env:MINICONDA_FILENAME
$client = new-object System.Net.WebClient
$client.DownloadFile($fileurl,  $filepath)

# Install miniconda

$install_args = "/InstallationType=AllUsers /AddToPath=1 /S /RegisterPython=1 /D=" + $env:PYTHON
Write-Host $filepath $install_args
Start-Process -Filepath $filepath -ArgumentList $install_args -Wait -Passthru
# At this point, conda should be installed, and the PATH appropriately updated
# Problem is, the updated PATH will only visible to SUBSEQUENT PowerShell sessions.
# In order to have PATH be correctly updated for the remainder of this PowerShell
# session, we need to manually update it. We do that below.
$env:PATH = "${env:PYTHON};${env:PYTHON}\Scripts;" + $env:PATH

# Configure conda

conda env create -n test_env -python=$env:PYTHON_VERSION
conda config --set always_yes true
conda config --add channels conda-forge cvxgrp anaconda
activate test_env
conda install $env:CONDA_DEPENDENCIES

# Install cvxpy

python setup.py install


