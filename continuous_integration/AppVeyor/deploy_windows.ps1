conda activate testenv
conda install --yes requests --no-update-dependencies

# Update the Windows binary hosted on Anaconda

cd continuous_integration
$UPDATE_CONDA = python -c "import versiongetter as vg; print(vg.update_conda('$env:PYTHON_VERSION','win'))"
cd ..
If ($UPDATE_CONDA -eq "True") {
    conda install --yes conda-build --no-update-dependencies
    conda install --yes anaconda-client --no-update-dependencies
    conda config --set anaconda_upload yes 
    conda config --add channels cvxgrp
    # ^ That line provides us with a limited version of SCS, for the sole purpose of building the conda
    # recipe. The standard suite of cvxpy tests will fail with this build, because this version of SCS
    # cant solve SDPs of order > 2.               
    conda build --token=$env:CONDA_UPLOAD_TOKEN --user=$env:CONDA_USER --python=$env:PYTHON_VERSION .
}

# Update the Windows wheel hosted in PyPI.

cd continuous_integration
$UPDATE_PYPI = python -c "import versiongetter as vg; print(vg.update_pypi_wheel('$env:PYTHON_VERSION','win','$env:PYPI_API_ENDPOINT'))"
cd ..
If ($UPDATE_PYPI -eq "True") {
    conda install --yes twine wheel readme_renderer --no-update-dependencies
    python setup.py bdist_wheel
    twine upload --repository-url $env:PYPI_SERVER dist/* -u $env:PYPI_USER -p $env:PYPI_PASSWORD
}
