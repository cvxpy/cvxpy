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
import copy

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.utilities import performance_utils as pu
from cvxpy.utilities.deterministic import unique_list


class Canonical(metaclass=abc.ABCMeta):
    """
    An interface for objects that can be canonicalized.
    """

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
        else:
            assert len(args) == len(self.args)
        data = self.get_data()
        if data is not None:
            return type(self)(*(args + data))
        else:
            return type(self)(*args)

    def _supports_cpp(self) -> bool:
        """
        Determines whether the current atom is implemented in C++. This method should be
        overridden in derived atom classes that are not implemented in C++.
        """
        return True

    def __copy__(self):
        """
        Called by copy.copy()
        Creates a shallow copy of the object, that is, the copied object refers to the same
        leaf nodes as the original object. Non-leaf nodes are recreated.
        Constraints keep their .id attribute, as it is used to propagate dual variables.

        Summary:
        ========
        Leafs:              Same object
        Constraints:        New object with same .id
        Other expressions:  New object with new .id
        """
        return self.copy()

    def __deepcopy__(self, memo):
        """
        Called by copy.deepcopy()
        Creates an independent copy of the object while maintaining the relationship between the
        nodes in the expression tree.
        """
        cvxpy_id = getattr(self, 'id', None)
        if cvxpy_id is not None and cvxpy_id in memo:
            return memo[cvxpy_id]
        else:
            with DefaultDeepCopyContextManager(self):  # Avoid infinite recursion
                new = copy.deepcopy(self, memo)
            if getattr(self, 'id', None) is not None:
                new_id = lu.get_id()
                new.id = new_id
            memo[cvxpy_id] = new
            return new

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

    @pu.compute_once
    def _aggregate_metrics(self) -> dict:
        """
        Aggregates and caches metrics for expression trees in self.args. So far
        metrics include the maximum dimensionality ('max_ndim') and whether
        all sub-expressions support C++ ('all_support_cpp').

        """
        max_ndim = self.ndim
        cpp_support = self._supports_cpp()

        for arg in self.args:
            max_ndim = max(max_ndim, arg._max_ndim())
            cpp_support = cpp_support and arg._all_support_cpp()

        metrics = {
            "max_ndim": max_ndim,
            "all_support_cpp": cpp_support
        }
        return metrics

    def _max_ndim(self) -> int:
        """The maximum number of dimensions of the sub-expression.
        """
        return self._aggregate_metrics()["max_ndim"]

    def _all_support_cpp(self) -> bool:
        """
        Returns True if all sub-expressions support C++, False otherwise.
        """
        return self._aggregate_metrics()["all_support_cpp"]

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()


_MISSING = object()


class DefaultDeepCopyContextManager:
    """
    override custom __deepcopy__ implementation and call copy.deepcopy's implementation instead
    """

    def __init__(self, item):
        self.item = item
        self.deepcopy = None

    def __enter__(self):
        self.deepcopy = getattr(self.item, '__deepcopy__', _MISSING)
        if self.deepcopy is not _MISSING:
            self.item.__deepcopy__ = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.deepcopy is not _MISSING:
            self.item.__deepcopy__ = self.deepcopy
            self.deepcopy = _MISSING
