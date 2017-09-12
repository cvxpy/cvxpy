import argparse
import re


# Captures row and column parameters; note the captured object should
# not be a keyword argument other than "cols" (hence the requirement that
# the captured group is followed by a comma, whitespace, or parentheses)
P_ROW_COL = r"(?:rows=)?(\w+),\s*(?:cols=)?(\w+)[\s,)]"

# A list of substitutions to make, with the first entry in each tuple the
# pattern and the second entry the substitution.
SUBST = [
    # The shape is a single argument in CVXPY 1.0 (either a tuple or an int)
    (r"Variable\(" + P_ROW_COL, r"Variable(shape=(\1, \2)"),
    (r"Bool\(" + P_ROW_COL, r"Variable(shape=(\1, \2), boolean=True"),
    (r"Int\(" + P_ROW_COL, r"Variable(shape=(\1, \2), integer=True"),
    (r"Parameter\(" + P_ROW_COL, r"Parameter(shape=(\1, \2)"),
    # Interpret 1D variables as 2D; code may depend upon 2D structure
    (r"Variable\(([^,)]+)\)", r"Variable(shape=(\1,1))"),
    (r"Bool\(([^,)]+)\)", r"Variable(shape=(\1,1), boolean=True)"),
    (r"Int\(([^,)]+)\)", r"Variable(shape=(\1,1), integer=True)"),
    (r"Parameter\(([^,)]+)\)", r"Parameter(shape=(\1,1))"),
    # Update atom names
    (r"sum_entries", "sum"),
    (r"max_entries", "cummax"),
    (r"max_elemwise", "max")
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Upgrade cvxpy code to version 1.0

        Usage:
            python cvxpy_upgrade.py --infile foo.py --outfile bar.py
        """)
    parser.add_argument("--infile", dest="input_file",
                        help="The name of the file to upgrade.",
                        required=True)
    parser.add_argument("--outfile", dest="output_file",
                        help="The output filename.",
                        required=True)
    args = parser.parse_args()
    with open(args.input_file, 'rU') as f:
        code = f.read()
    for pattern, subst in SUBST:
        code = re.sub(pattern, subst, code)
    with open(args.output_file, 'w') as f:
        f.write(code)
