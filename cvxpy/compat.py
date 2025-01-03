import scipy.version

_scipy_version = tuple(int(v) for v in scipy.version.short_version.split('.'))

if _scipy_version < (1, 15, 0):
    class scipy_coo_array_compat:
        def __init__(self, data_coords, dtype, shape):
            self.shape = shape
            self.dtype = dtype
            self.data, self.coords = data_coords
    scipy_coo_array = scipy_coo_array_compat
    scipy_coo_array_name = 'cvxpy.compat.scipy_coo_array'
else:
    scipy_coo_array = scipy.sparse.coo_array
    scipy_coo_array_name = 'scipy.sparse.coo_array'
