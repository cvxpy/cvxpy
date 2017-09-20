import argparse
import re

# A simple script to help upgrade pre-1.0 source files to version 1.0.
# This script does nothing more than find-and-replace substitutions;
# conversions may fail silently. Users should make sure to diff output files
# with the input files in order to catch and fix any errors.

# Captures row and column parameters; note the captured object should
# not be a keyword argument other than "cols" (hence the requirement that
# the second captured group is not followed by an equals operator)
P_ROW_COL = r"(?:rows=)?(\w+),\s*(?:cols\s*=)?(\w+)\s*(?![\s\w=])"

# A list of substitutions to make, with the first entry in each tuple the
# pattern and the second entry the substitution.
SUBST = [
    # The shape is a single argument in CVXPY 1.0 (either a tuple or an int)
    (r"Variable\(" + P_ROW_COL, r"Variable(shape=(\1, \2)"),
    (r"NonNegative\(" + P_ROW_COL, r"Variable(shape=(\1, \2), nonneg=True"),
    (r"Bool\(" + P_ROW_COL, r"Variable(shape=(\1, \2), boolean=True"),
    (r"Int\(" + P_ROW_COL, r"Variable(shape=(\1, \2), integer=True"),
    (r"Parameter\(" + P_ROW_COL, r"Parameter(shape=(\1, \2)"),
    # Interpret 1D variables as 2D; code may depend upon 2D structure
    (r"Variable\((\w+)\)", r"Variable(shape=(\1,1))"),
    (r"NonNegative\((\w+)\)", r"Variable(shape=(\1,1), nonneg=True)"),
    (r"Bool\((\w+)\)", r"Variable(shape=(\1,1), boolean=True)"),
    (r"Int\((\w+)\)", r"Variable(shape=(\1,1), integer=True)"),
    (r"Parameter\((\w+)\)", r"Parameter(shape=(\1,1))"),
    (r"Parameter\((\w+), value=(\w+)\)", r"Parameter(shape=(\1,1), value=\2)"),
    # Symmetric and PSD
    (r"Symmetric\((\w+)\)", r"Variable(shape=(\1,\1), symmetric=True)"),
    (r"Semidef\((\w+)\)", r"Variable(shape=(\1,\1), PSD=True)"),
    (r"semidefinite\((\w+)\)", r"Variable(shape=(\1,\1), PSD=True)"),
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
