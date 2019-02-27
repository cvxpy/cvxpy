conda activate testenv
conda install --yes requests

# Update the Windows binary hosted on Anaconda

cd continuous_integration
$UPDATE_CONDA = python -c "import versiongetter as vg; print(vg.should_update_conda('$env:PYTHON_VERSION','win'))"
cd ..
If ($UPDATE_CONDA -eq "True") {
    conda install --yes conda-build
    conda install --yes anaconda-client
    conda config --set anaconda_upload yes
    conda build --token=$env:CONDA_UPLOAD_TOKEN --user=$env:CONDA_USER --python=$env:PYTHON_VERSION .
}

# The pypi uploads will look something like the lines below (once implemented)
#
# cd continuous_integration
# $UPDATE_PYPI = python -c "import versiongetter as vg; print(vg.should_update_pypi('$env:PYTHON_VERSION','win'))"
# cd ..
#
# If ($UPDATE_PYPI -eq "True") {
#     conda install --yes twine
#     conda install --yes wheel
#     python setup.py sdist bdist_wheel
#     twine upload --repository-url $env:PYPI_SERVER dist/* -u $env:PYPI_USER -p $PYPI_PASSWORD
# }
