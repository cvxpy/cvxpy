conda activate testenv
conda install --yes requests

# Update the Windows wheel hosted in PyPI.

cd continuous_integration
$UPDATE_PYPI = python -c "import versiongetter as vg; print(vg.update_pypi_wheel('$env:PYTHON_VERSION','win','$env:PYPI_API_ENDPOINT'))"
cd ..
If ($UPDATE_PYPI -eq "True") {
    conda install --yes twine wheel
    conda install -c conda-forge readme_renderer
    python setup.py bdist_wheel
    twine upload --repository-url $env:PYPI_SERVER dist/* -u $env:PYPI_USER -p $env:PYPI_PASSWORD
}
