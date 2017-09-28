"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import abc
from cvxpy.utilities import performance_utils as pu


class Canonical(object):
    """
    An interface for objects that can be canonicalized.
    """

    __metaclass__ = abc.ABCMeta

    @property
    def expr(self):
        if not len(self.args) == 1:
            raise ValueError("'expr' is ambiguous, there should be only one argument")
        return self.args[0]

    @pu.lazyprop
    def canonical_form(self):
        """The graph implementation of the object stored as a property.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        return self.canonicalize()

    def variables(self):
        """Returns all the variables present in the arguments.
        """
        var_list = []
        for arg in self.args:
            var_list += arg.variables()
        # Remove duplicates.
        return list(set(var_list))

    def parameters(self):
        """Returns all the parameters present in the arguments.
        """
        param_list = []
        for arg in self.args:
            param_list += arg.parameters()
        # Remove duplicates.
        return list(set(param_list))

    def constants(self):
        """Returns all the constants present in the arguments.
        """
        const_list = []
        const_dict = {}
        for arg in self.args:
            const_list += arg.constants()
        # Remove duplicates:
        const_dict = {id(constant): constant for constant in const_list}
        return list(const_dict.values())

    def tree_copy(self, id_objects={}):
        new_args = []
        for arg in self.args:
            if isinstance(arg, list):
                arg_list = []
                for elem in arg:
                    arg_list += [elem.tree_copy(id_objects)]
                new_args.append(arg_list)
            else:
                new_args += [arg.tree_copy(id_objects)]
        return self.copy(args=new_args, id_objects=id_objects)

    def copy(self, args=None, id_objects={}):
        """Returns a shallow copy of the object.

        Used to reconstruct an object tree.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the object. If args=None, use the
            current args of the object.

        Returns
        -------
        Expression
        """
        if id(self) in id_objects:
            return id_objects[id(self)]
        if args is None:
            args = self.args
        data = self.get_data()
        if data is not None:
            return type(self)(*(args + data))
        else:
            return type(self)(*args)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return None

    def atoms(self):
        """Returns all the atoms present in the args.

        Returns
        -------
        list
        """
        atom_list = []
        for arg in self.args:
            atom_list += arg.atoms()
        # Remove duplicates.
        return list(set(atom_list))
