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
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from cvxpy.constraints.constraint import Constraint

import numpy as np

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.settings as s
from cvxpy import interface as intf
from cvxpy import utilities as u
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.utilities import performance_utils as perf
from cvxpy.utilities.deterministic import unique_list


class Atom(Expression):
    """ Abstract base class for atoms. """
    _allow_complex = False
    # args are the expressions passed into the Atom constructor.

    def __init__(self, *args) -> None:
        self.id = lu.get_id()
        # Throws error if args is empty.
        if len(args) == 0:
            raise TypeError(
                "No arguments given to %s." % self.__class__.__name__
            )
        # Convert raw values to Constants.
        self.args = [Atom.cast_to_const(arg) for arg in args]
        self.validate_arguments()
        self._shape = self.shape_from_args()
        if not s.ALLOW_ND_EXPR and len(self._shape) > 2:
            raise ValueError("Atoms must be at most 2D.")

    def name(self) -> str:
        """Returns the string representation of the function call.
        """
        if self.get_data() is None:
            data = []
        else:
            data = [str(elem) for elem in self.get_data()]
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join([arg.name() for arg in self.args] + data))

    def validate_arguments(self) -> None:
        """Raises an error if the arguments are invalid.
        """
        if not self._allow_complex and any(arg.is_complex() for arg in self.args):
            raise ValueError(
                "Arguments to %s cannot be complex." % self.__class__.__name__
            )

    @abc.abstractmethod
    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the expression.
        """
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @abc.abstractmethod
    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        raise NotImplementedError()

    @perf.compute_once
    def is_nonneg(self) -> bool:
        """Is the expression nonnegative?
        """
        return self.sign_from_args()[0]

    @perf.compute_once
    def is_nonpos(self) -> bool:
        """Is the expression nonpositive?
        """
        return self.sign_from_args()[1]

    @perf.compute_once
    def is_imag(self) -> bool:
        """Is the expression imaginary?
        """
        # Default is false.
        return False

    @perf.compute_once
    def is_complex(self) -> bool:
        """Is the expression complex valued?
        """
        # Default is false.
        return False

    @abc.abstractmethod
    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        raise NotImplementedError()

    def is_atom_affine(self) -> bool:
        """Is the atom affine?
        """
        return self.is_atom_concave() and self.is_atom_convex()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_atom_quasiconvex(self) -> bool:
        """Is the atom quasiconvex?
        """
        return self.is_atom_convex()

    def is_atom_quasiconcave(self) -> bool:
        """Is the atom quasiconcave?
        """
        return self.is_atom_concave()

    def is_atom_log_log_affine(self) -> bool:
        """Is the atom log-log affine?
        """
        return self.is_atom_log_log_concave() and self.is_atom_log_log_convex()

    @abc.abstractmethod
    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        raise NotImplementedError()

    @perf.compute_once
    def is_convex(self) -> bool:
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

    @perf.compute_once
    def is_concave(self) -> bool:
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

    def is_dpp(self, context='dcp') -> bool:
        """The expression is a disciplined parameterized expression.
        """
        if context.lower() == 'dcp':
            return self.is_dcp(dpp=True)
        elif context.lower() == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError('Unsupported context ', context)

    @perf.compute_once
    def is_log_log_convex(self) -> bool:
        """Is the expression log-log convex?
        """
        # Verifies DGP composition rule.
        if self.is_log_log_constant():
            return True
        elif self.is_atom_log_log_convex():
            for idx, arg in enumerate(self.args):
                if not (arg.is_log_log_affine() or
                        (arg.is_log_log_convex() and self.is_incr(idx)) or
                        (arg.is_log_log_concave() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @perf.compute_once
    def is_log_log_concave(self) -> bool:
        """Is the expression log-log concave?
        """
        # Verifies DGP composition rule.
        if self.is_log_log_constant():
            return True
        elif self.is_atom_log_log_concave():
            for idx, arg in enumerate(self.args):
                if not (arg.is_log_log_affine() or
                        (arg.is_log_log_concave() and self.is_incr(idx)) or
                        (arg.is_log_log_convex() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @perf.compute_once
    def _non_const_idx(self) -> List[int]:
        return [i for i, arg in enumerate(self.args) if not arg.is_constant()]

    @perf.compute_once
    def _is_real(self) -> bool:
        # returns true if this atom is a real function:
        #   the atom must have exactly one argument that is not a constant
        #   that argument must be a scalar
        #   the output must be a scalar
        non_const = self._non_const_idx()
        return (self.is_scalar() and len(non_const) == 1 and
                self.args[non_const[0]].is_scalar())

    @perf.compute_once
    def is_quasiconvex(self) -> bool:
        """Is the expression quaisconvex?
        """
        from cvxpy.atoms.max import max as max_atom

        # Verifies the DQCP composition rule.
        if self.is_convex():
            return True
        if type(self) in (cvxtypes.maximum(), max_atom):
            return all(arg.is_quasiconvex() for arg in self.args)
        non_const = self._non_const_idx()
        if self._is_real() and self.is_incr(non_const[0]):
            return self.args[non_const[0]].is_quasiconvex()
        if self._is_real() and self.is_decr(non_const[0]):
            return self.args[non_const[0]].is_quasiconcave()
        if self.is_atom_quasiconvex():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_convex() and self.is_incr(idx)) or
                        (arg.is_concave() and self.is_decr(idx))):
                    return False
            return True
        return False

    @perf.compute_once
    def is_quasiconcave(self) -> bool:
        """Is the expression quasiconcave?
        """
        from cvxpy.atoms.min import min as min_atom

        # Verifies the DQCP composition rule.
        if self.is_concave():
            return True
        if type(self) in (cvxtypes.minimum(), min_atom):
            return all(arg.is_quasiconcave() for arg in self.args)
        non_const = self._non_const_idx()
        if self._is_real() and self.is_incr(non_const[0]):
            return self.args[non_const[0]].is_quasiconcave()
        if self._is_real() and self.is_decr(non_const[0]):
            return self.args[non_const[0]].is_quasiconvex()
        if self.is_atom_quasiconcave():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_concave() and self.is_incr(idx)) or
                        (arg.is_convex() and self.is_decr(idx))):
                    return False
            return True
        return False

    def canonicalize(self):
        """Represent the atom as an affine objective and conic constraints.
        """
        # Constant atoms are treated as a leaf.
        if self.is_constant() and not self.parameters():
            # Non-parameterized expressions are evaluated immediately.
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
                                                                self.shape,
                                                                data)
            return graph_obj, constraints + graph_constr

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List['Constraint']]:
        """Reduces the atom to an affine expression and list of constraints.

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
        raise NotImplementedError()

    @property
    def value(self):
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()

    def _value_impl(self):
        # shapes with 0's dropped in presolve.
        if 0 in self.shape:
            result = np.array([])
        else:
            arg_values = []
            for arg in self.args:
                # A argument without a value makes all higher level
                # values None.
                # But if the atom is constant with non-constant
                # arguments it doesn't depend on its arguments,
                # so it isn't None.
                arg_val = arg._value_impl()
                if arg_val is None and not self.is_constant():
                    return None
                else:
                    arg_values.append(arg_val)
            result = self.numeric(arg_values)
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
        raise NotImplementedError()

    @property
    def domain(self) -> List['Constraint']:
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        return self._domain() + [con for arg in self.args for con in arg.domain]

    def _domain(self) -> List['Constraint']:
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

    def atoms(self) -> List['Atom']:
        """A list of the atom types present amongst this atom's arguments.
        """
        atom_list = []
        for arg in self.args:
            atom_list += arg.atoms()
        return unique_list(atom_list + [type(self)])
