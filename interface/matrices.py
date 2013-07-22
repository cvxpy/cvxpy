import cvxopt
import numbers
import numpy
TARGET_MATRIX = cvxopt.matrix

# Convert an arbitrary value into a matrix of type TARGET_MATRIX.
def const_to_matrix(value):
    # if isinstance(value, numbers.Number): TODO promotion
    #     return value
    # elif isinstance(value, cvxopt.matrix) or isinstance(value, list):
    return TARGET_MATRIX(value, tc='d')

# Return an identity matrix.
def identity(size):
    matrix = zeros(size, size)
    for i in range(size):
        matrix[i,i] = 1
    return matrix

# Return a matrix with all 0's.
def zeros(rows, cols):
    return scalar_matrix(0, rows, cols)

# Return a matrix with all 1's.
def ones(rows, cols):
    return scalar_matrix(1, rows, cols)

# A matrix with all entries equal to the given scalar value.
def scalar_matrix(value, rows, cols):
    return TARGET_MATRIX(value, (rows,cols), tc='d')

# Copy the block into the matrix at the given offset.
# If the block is a vector coerced into a matrix, promote it.
# If the block is a matrix coerced into a vector, vectorize it.
# If the block is a scalar, promote it.
def block_copy(matrix, block, vert_offset, horiz_offset, rows, cols):
    if is_scalar(block):
        block = scalar_matrix(scalar_value(block), rows, cols)
    elif is_vector(block) and cols > 1:
        block = TARGET_MATRIX(list(block), (rows, cols), tc='d')
    elif not is_vector(block) and cols == 1:
        block = TARGET_MATRIX(list(block), tc='d')
    matrix[vert_offset:(rows+vert_offset), horiz_offset:(horiz_offset+cols)] = block

# Get the dimensions of the constant.
def size(constant):
    if isinstance(constant, numbers.Number):
        return (1,1)
    elif isinstance(constant, list):
        if len(constant) == 0:
            return (0,0)
        elif isinstance(constant[0], numbers.Number): # Vector
            return (len(constant),1)
        else: # Matrix
            return (len(constant[0]),len(constant))
    elif isinstance(constant, cvxopt.matrix):
        return constant.size
    elif isinstance(constant, numpy.ndarray):
        return constant.shape

# Is the constant a vector?
def is_vector(constant):
    return size(constant)[1] == 1

# Is the constant a scalar?
def is_scalar(constant):
    return size(constant) == (1,1)

# Get the value of the passed constant, interpreted as a scalar.
def scalar_value(constant):
    assert is_scalar(constant)
    if isinstance(constant, numbers.Number):
        return constant
    elif isinstance(constant, list):
        return constant[0]
    elif isinstance(constant, cvxopt.matrix):
        return constant[0,0]