def are_args_affine(constraints):
    return all(arg.is_affine() for constr in constraints
               for arg in constr.args)
