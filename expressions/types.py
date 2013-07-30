# Utility functions to solve circular imports.
def constant():
    import constant
    return constant.Constant

def variable():
    import variable
    return variable.Variable

def parameter():
    import parameter
    return parameter.Parameter