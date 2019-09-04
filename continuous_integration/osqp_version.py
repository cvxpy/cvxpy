#!/usr/bin/env python
"""
Print the library version of OSQP and compare to pkg_resource
"""
import ctypes
import pkg_resources
import osqp


dl = ctypes.CDLL(osqp._osqp.__file__)
dl.osqp_version.restype = ctypes.c_char_p
print("dynamic library version:", dl.osqp_version().decode())
print("pkg_resources version:  ", pkg_resources.get_distribution("osqp").version)
