# Download miniconda

$miniconda_url = "https://repo.continuum.io/miniconda/"
$fileurl = $miniconda_url + $env:MINICONDA_FILENAME
$filepath = $pwd.Path + "\" + $env:MINICONDA_FILENAME
$client = new-object System.Net.WebClient
$client.DownloadFile($fileurl,  $filepath)

# Install miniconda

$install_args = "/InstallationType=AllUsers /S /RegisterPython=1 /D=" + $env:PYTHON
Write-Host $filepath $install_args
Start-Process -Filepath $filepath -ArgumentList $install_args -Wait -Passthru
# The conda install doesn't work well when called from PowerShell.
# We need to set some environment variables.
$dir_to_add = "${env:PYTHON};${env:PYTHON}\Scripts;"
$env:PATH = $dir_to_add + $env:PATH
echo "The PATH environment variable is"
echo $env:PATH

# Configure miniconda

conda create -n $env:ENV_NAME --yes python=$env:PYTHON_VERSION mkl pip nose numpy scipy
activate $env:ENV_NAME
# The conda activation doesn't work well from PowerShell.
# We need to update environment variables.
$base_env_dir = "${env:PYTHON}\envs\$env:ENV_NAME"
$env:PATH = "$base_env_dir;$base_env_dir\Scripts;$base_env_dir\Library\bin;" + $env:PATH
conda install -c conda-forge --yes lapack
conda install -c cvxgrp --yes ecos scs multiprocess
conda install -c anaconda --yes flake8

# Install cvxpy

python setup.py install


