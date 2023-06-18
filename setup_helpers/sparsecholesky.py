import platform

from pybind11.setup_helpers import Pybind11Extension


def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != "windows" else ""


sparsecholesky = Pybind11Extension("_cvxpy_sparsecholesky",
      ["cvxpy/utilities/cpp/sparsecholesky/main.cpp"],
      define_macros=[('VERSION_INFO', "0.0.1")],
       extra_compile_args=[
           '-O3',
           '-std=c++11',
           '-Wall',
           '-pedantic',
           not_on_windows('-Wextra'),
           not_on_windows('-Wno-unused-parameter'),
       ],
       extra_link_args=['-O3'],
)
