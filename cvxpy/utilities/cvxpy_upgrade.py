import argparse
import re

# A simple script to help upgrade pre-1.0 source files to version 1.0.
# This script does nothing more than find-and-replace substitutions;
# conversions may fail silently. Users should make sure to diff output files
# with the input files in order to catch and fix any errors.

# Captures row and column parameters; note the captured object should
# not be a keyword argument other than "cols" (hence the requirement that
# the second captured group is not followed by an equals operator)
TOK = r"[\w*+-/\s]"
P_ROW_COL = r"(?:rows=)?({0}+),\s*(?:cols\s*=)?({0}+)\s*(?![\s\w=])".format(TOK)

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
    (r"Variable\(({0}+)\)".format(TOK), r"Variable(shape=(\1,1))"),
    (r"NonNegative\(({0}+)\)".format(TOK), r"Variable(shape=(\1,1), nonneg=True)"),
    (r"Bool\(({0}+)\)".format(TOK), r"Variable(shape=(\1,1), boolean=True)"),
    (r"Int\(({0}+)\)".format(TOK), r"Variable(shape=(\1,1), integer=True)"),
    (r"Parameter\(({0}+)\)".format(TOK), r"Parameter(shape=(\1,1))"),
    (r"Parameter\(({0}+), value=({0}+)\)".format(TOK), r"Parameter(shape=(\1,1), value=\2)"),
    # Symmetric and PSD
    (r"Symmetric\(({0}+)\)".format(TOK), r"Variable(shape=(\1,\1), symmetric=True)"),
    (r"Semidef\(({0}+)\)".format(TOK), r"Variable(shape=(\1,\1), PSD=True)"),
    (r"semidefinite\(({0}+)\)".format(TOK), r"Variable(shape=(\1,\1), PSD=True)"),
    # Update atom names
    (r"sum_entries", "sum"),
    (r"mul_elemwise", "multiply"),
    (r"max_entries", "max"),
    (r"min_entries", "min"),
    (r"max_elemwise", "maximum"),
    (r"min_elemwise", "minimum"),
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
