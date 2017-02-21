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


from .. import utilities as u
from .. import interface as intf
from ..expressions.constants import Constant, CallbackParam
from ..expressions.expression import Expression
import abc
import numpy as np
from fastcache import clru_cache


class Atom(Expression):
    """ Abstract base class for atoms. """
    __metaclass__ = abc.ABCMeta
    # args are the expressions passed into the Atom constructor.

    def __init__(self, *args):
        # Throws error if args is empty.
        if len(args) == 0:
            raise TypeError(
                "No arguments given to %s." % self.__class__.__name__
            )
        # Convert raw values to Constants.
        self.args = [Atom.cast_to_const(arg) for arg in args]
        self.validate_arguments()
        self._size = self.size_from_args()

    def name(self):
        """Returns the string representation of the function call.
        """
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join([arg.name() for arg in self.args]))

    def validate_arguments(self):
        """Raises an error if the arguments are invalid.
        """
        pass

    @abc.abstractmethod
    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return NotImplemented

    @property
    def size(self):
        return self._size

    @abc.abstractmethod
    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return NotImplemented

    @clru_cache(maxsize=100)
    def is_positive(self):
        """Is the expression positive?
        """
        return self.sign_from_args()[0]

    @clru_cache(maxsize=100)
    def is_negative(self):
        """Is the expression negative?
        """
        return self.sign_from_args()[1]

    @abc.abstractmethod
    def is_atom_convex(self):
        """Is the atom convex?
        """
        return NotImplemented

    @abc.abstractmethod
    def is_atom_concave(self):
        """Is the atom concave?
        """
        return NotImplemented

    def is_atom_affine(self):
        """Is the atom affine?
        """
        return self.is_atom_concave() and self.is_atom_convex()

    @abc.abstractmethod
    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return NotImplemented

    @abc.abstractmethod
    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return NotImplemented

    @clru_cache(maxsize=100)
    def is_convex(self):
        """Is the expression convex?
        """
        # Applies DCP composition rule.
        if self.is_constant():
            return True
        elif self.is_atom_convex():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_convex() and self.is_incr(idx)) or
                        (arg.is_concave() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @clru_cache(maxsize=100)
    def is_concave(self):
        """Is the expression concave?
        """
        # Applies DCP composition rule.
        if self.is_constant():
            return True
        elif self.is_atom_concave():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_concave() and self.is_incr(idx)) or
                        (arg.is_convex() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    def canonicalize(self):
        """Represent the atom as an affine objective and conic constraints.
        """
        # Constant atoms are treated as a leaf.
        if self.is_constant():
            # Parameterized expressions are evaluated later.
            if self.parameters():
                rows, cols = self.size
                param = CallbackParam(self, rows, cols)
                return param.canonical_form
            # Non-parameterized expressions are evaluated immediately.
            else:
                return Constant(self.value).canonical_form
        else:
            arg_objs = []
            constraints = []
            for arg in self.args:
                obj, constr = arg.canonical_form
                arg_objs.append(obj)
                constraints += constr
            # Special info required by the graph implementation.
            data = self.get_data()
            graph_obj, graph_constr = self.graph_implementation(arg_objs,
                                                                self.size,
                                                                data)
            return (graph_obj, constraints + graph_constr)

    @abc.abstractmethod
    def graph_implementation(self, arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

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
        return NotImplemented

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

    @property
    def value(self):
        # Catch the case when the expression is known to be
        # zero through DCP analysis.
        if self.is_zero():
            result = intf.DEFAULT_INTF.zeros(*self.size)
        else:
            arg_values = []
            for arg in self.args:
                # A argument without a value makes all higher level
                # values None.
                # But if the atom is constant with non-constant
                # arguments it doesn't depend on its arguments,
                # so it isn't None.
                arg_val = arg.value
                if arg_val is None and not self.is_constant():
                    return None
                else:
                    arg_values.append(arg_val)
            result = self.numeric(arg_values)

        # Reduce to a scalar if possible.
        if intf.size(result) == (1, 1):
            return intf.scalar_value(result)
        else:
            return result

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.
        None indicates variable values unknown or outside domain.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        # Short-circuit to all zeros if known to be constant.
        if self.is_constant():
            return u.grad.constant_grad(self)

        # Returns None if variable values not supplied.
        arg_values = []
        for arg in self.args:
            if arg.value is None:
                return u.grad.error_grad(self)
            else:
                arg_values.append(arg.value)

        # A list of gradients w.r.t. arguments
        grad_self = self._grad(arg_values)
        # The Chain rule.
        result = {}
        for idx, arg in enumerate(self.args):
            # A dictionary of gradients w.r.t. variables
            # Partial argument / Partial x.
            grad_arg = arg.grad
            for key in grad_arg:
                # None indicates gradient is not defined.
                if grad_arg[key] is None or grad_self[idx] is None:
                    result[key] = None
                else:
                    D = grad_arg[key]*grad_self[idx]
                    # Convert 1x1 matrices to scalars.
                    if not np.isscalar(D) and D.shape == (1, 1):
                        D = D[0, 0]

                    if key in result:
                        result[key] += D
                    else:
                        result[key] = D

        return result

    @abc.abstractmethod
    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return NotImplemented

    @property
    def domain(self):
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        return self._domain() + [con for arg in self.args for con in arg.domain]

    def _domain(self):
        """Returns constraints describing the domain of the atom.
        """
        # Default is no constraints.
        return []

    @staticmethod
    def numpy_numeric(numeric_func):
        """Wraps an atom's numeric function that requires numpy ndarrays as input.
           Ensures both inputs and outputs are the correct matrix types.
        """

        def new_numeric(self, values):
            interface = intf.DEFAULT_INTF
            values = [interface.const_to_matrix(v, convert_scalars=True)
                      for v in values]
            result = numeric_func(self, values)
            return intf.DEFAULT_INTF.const_to_matrix(result)
        return new_numeric
