# Assume we're in the same directory as setup.py

cd cvxpy/cvxcore
swig -Isrc -c++ -python python/cvxcore.i
cd ..
cd ..
pip install -e .
