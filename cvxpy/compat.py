import scipy.version

_scipy_version = tuple(int(v) for v in scipy.version.short_version.split('.'))

if _scipy_version < (1, 15, 0):
    class scipy_coo_array_compat:
        """
        Supported Constructors:
            ((data, coords), dtype, shape)
            (sparse_matrix)
        """
        def __init__(self, data_coords, dtype=None, shape=None):
            if dtype is None and shape is None:
                shape = data_coords.shape
                dtype = data_coords.dtype
                coo = data_coords.tocoo()
                self.data = coo.data
                self.coords = coo.coords
            else:
                self.shape = shape
                self.dtype = dtype
                self.data, self.coords = data_coords
    scipy_coo_array = scipy_coo_array_compat
    scipy_coo_array_name = 'cvxpy.compat.scipy_coo_array'
else:
    scipy_coo_array = scipy.sparse.coo_array
    scipy_coo_array_name = 'scipy.sparse.coo_array'
