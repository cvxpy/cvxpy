SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE = """
The second argument has type cvxpy.Expression. However, that is not allowed.
Most likely, you want to call this atom by using cvxpy.hstack to combine the
two arguments into a vector.
"""
