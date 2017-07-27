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

import abc
from cvxpy.utilities import performance_utils as pu


class Canonical(object):
    """
    An interface for objects that can be canonicalized.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        return NotImplemented

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

    def copy(self, args=None):
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
