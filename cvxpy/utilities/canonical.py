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

import abc

from cvxpy.utilities import performance_utils as pu
from cvxpy.utilities.deterministic import unique_list


class Canonical:
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

    # TODO(akshayka): some code relies on these not being cached, figure out
    # which code and fix it
    def variables(self):
        """Returns all the variables present in the arguments.
        """
        return unique_list(
            [var for arg in self.args for var in arg.variables()])

    def parameters(self):
        """Returns all the parameters present in the arguments.
        """
        return unique_list(
            [param for arg in self.args for param in arg.parameters()])

    def constants(self):
        """Returns all the constants present in the arguments.
        """
        return unique_list(
            [const for arg in self.args for const in arg.constants()])

    def tree_copy(self, id_objects=None):
        new_args = []
        for arg in self.args:
            if isinstance(arg, list):
                arg_list = [elem.tree_copy(id_objects) for elem in arg]
                new_args.append(arg_list)
            else:
                new_args.append(arg.tree_copy(id_objects))
        return self.copy(args=new_args, id_objects=id_objects)

    def copy(self, args=None, id_objects=None):
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
        id_objects = {} if id_objects is None else id_objects
        if id(self) in id_objects:
            return id_objects[id(self)]
        if args is None:
            args = self.args
        data = self.get_data()
        if data is not None:
            return type(self)(*(args + data))
        else:
            return type(self)(*args)

    def get_data(self) -> None:
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
        # Remove duplicates.
        return unique_list(atom for arg in self.args for atom in arg.atoms())
