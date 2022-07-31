def unique_list(duplicates_list):
    """
    Return unique list preserving the order.
    https://stackoverflow.com/a/480227
    """
    used = set()
    unique = [x for x in duplicates_list if not (x in used or used.add(x))]
    return unique


def unique_expressions(duplicates_list):
    """
    Return unique list of expressions preserving the order.
    This function only considers the expression.id and thus is invariant under deepcopy of a
    subset of list elements.
    """
    used = set()
    unique = [x for x in duplicates_list if not (x.id in used or used.add(x.id))]
    return unique
