def unique_list(duplicates_list):
    """Return unique list preserving the order.
    https://stackoverflow.com/a/37163210/9753175
    """
    used = set()
    unique = [x for x in duplicates_list
              if x not in used and
              (used.add(x) or True)]
    return unique
