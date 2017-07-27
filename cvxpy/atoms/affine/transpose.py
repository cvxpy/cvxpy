"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.atoms.affine.affine_atom import AffAtom
import cvxpy.lin_ops.lin_utils as lu


class transpose(AffAtom):
    """ Matrix transpose. """
    # The string representation of the atom.

    def name(self):
        return "%s.T" % self.args[0]

    # Returns the transpose of the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return values[0].T

    def size_from_args(self):
        """Returns the shape of the transpose expression.
        """
        rows, cols = self.args[0].size
        return (cols, rows)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Create a new variable equal to the argument transposed.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.transpose(arg_objs[0]), [])
