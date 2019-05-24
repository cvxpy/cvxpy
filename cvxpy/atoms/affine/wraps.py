"""
Copyright 2013 Steven Diamond

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


class Wrap(AffAtom):
    """A no-op wrapper to assert properties.
    """
    def __init__(self, arg):
        return super(Wrap, self).__init__(arg)

    def is_atom_log_log_convex(self):
        return True

    def is_atom_log_log_concave(self):
        return True

    def numeric(self, values):
        """ Returns input.
        """
        return values[0]

    def shape_from_args(self):
        """Shape of input.
        """
        return self.args[0].shape

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Stack the expressions horizontally.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (arg_objs[0], [])


class psd_wrap(Wrap):
    """Asserts argument is PSD.
    """

    def is_psd(self):
        return True
