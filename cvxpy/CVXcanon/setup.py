from setuptools import setup, Extension


class get_numpy_include(object):
    """Returns Numpy's include path with lazy import.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


canon = Extension(
    '_CVXcanon',
    sources=['src/CVXcanon.cpp', 'src/LinOpOperations.cpp', 'src/python/CVXcanon_wrap.cpp'],
    include_dirs=['src/', 'src/python/', 'include/Eigen', get_numpy_include()]
)


setup(
    name='CVXcanon',
    version="0.1.1",
    setup_requires=['numpy'],
    author='Jack Zhu, John Miller, Paul Quigley',
    author_email='jackzhu@stanford.edu, millerjp@stanford.edu, piq93@stanford.edu',
    ext_modules=[canon],
    package_dir={'': 'src/python'},
    py_modules=['canonInterface', 'CVXcanon', '_version__'],
    description='A low-level library to perform the matrix building step in cvxpy, a convex optimization modeling software.',
    license='GPLv3',
    url='https://github.com/jacklzhu/CVXcanon',
    install_requires=[
        'numpy',
        'scipy',
    ]
)
