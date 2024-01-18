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

import numpy as np
import torch

import cvxpy.atoms as atoms
import cvxpy.atoms.affine as affine
import cvxpy.atoms.elementwise as elementwise
from cvxpy.tests.base_test import BaseTest


class TestTorchNumeric(BaseTest):
    """ Unit tests for the atoms torch_numeric. """

    def _assert_torch_numeric(self, atom, tests_flags=(True, True, True), repetitions=1,
                              dtype=torch.float64, *args):
        """
        This funciton tests if atom.torch_numeric is correct.

        Args:
            atom
                Atom to check
            test_flags
                A boolean tuple:
                    If i is True    : Test on this
                    ===============================
                    0               : self.arr_np
                    1               : self.arr_np
                    2               : self.sqr_np
                    3               : self.cmp_np (not included in the standard tests)
            repetitions
                How many times to duplicate the input
            dtype
                What dtype to use with torch.tensor
            *args
                To create a custom test, pass tuples where each element is 0/1/2 corresponding to
                which type of element is the next element.
                For example, for a test where the first element is the 2x3 matrix
                and the second element is the 1D vector, pass (1,0).

        Comments:
            You may notice that during the test, an object is initiated with a dummy variable
            (often None). That is because to test torch_numeric, this value is often ignored,
            and only the passed values are used.
            
        """

        def perform_test(atom, in_np, in_tch):
            """
            This is an internal function that performs a test and compares the result between
            the np version and the torch version.
            """
            res_np = atom.numeric(in_np)
            res_tch = atom.torch_numeric(in_tch)
            if type(res_tch) is torch.Tensor:
                res_tch = res_tch.detach().numpy()
            assert np.allclose(res_np, res_tch, equal_nan=True)

        assert hasattr(atom, "torch_numeric"), f"Atom {atom} has no torch_numeric function."
        #Perform predefined tests
        for i, curr_test in enumerate(tests_flags):
            if not curr_test:
                continue
            item_np = self.test_dict[i]
            item_tch = torch.tensor(item_np, dtype=dtype)
            in_np = [item_np]*repetitions
            in_tch = [item_tch]*repetitions
            perform_test(atom=atom, in_np=in_np, in_tch=in_tch)
            
        #Perform custom tests
        for curr_test in args:
            in_np = []
            in_tch = []
            for inp in curr_test:
                item_np = self.test_dict[inp] 
                in_np.append(item_np)
                in_tch.append(torch.tensor(item_np, dtype=dtype))
            perform_test(atom=atom, in_np=in_np, in_tch=in_tch)

    def setUp(self) -> None:
        np.random.seed(1234)
        self.arr_np = np.array([1, -1, 0])
        self.mat_np = np.array([[1,-1,0],[0,1,2]])
        self.sqr_np = np.random.randn(3,3)
        self.cmp_np = np.array([[2+3.j, 5+8.j, 0-2.j],[8+0.j, 1-3.j, 2+2.j]])

        self.test_dict = {
            0: self.arr_np,
            1: self.mat_np,
            2: self.sqr_np,
            3: self.cmp_np,
            }

    def test_atoms(self):
        self._assert_torch_numeric(atoms.cummax(None))
        self._assert_torch_numeric(atoms.dist_ratio(None, 10, 5))
        self._assert_torch_numeric(atoms.dotsort(None, 4), repetitions=2)
        self._assert_torch_numeric(atoms.eye_minus_inv(np.eye(3)), (True, False, True))
        self._assert_torch_numeric(atoms.geo_mean(None))
        self._assert_torch_numeric(atoms.gmatmul(np.ones(3),np.ones(3)), (True, False, True))
        self._assert_torch_numeric(atoms.length(np.ones(3)))
        self._assert_torch_numeric(atoms.log_det(np.eye(3)), (False, False, True))
        self._assert_torch_numeric(atoms.log_sum_exp(None))
        self._assert_torch_numeric(atoms.MatrixFrac(np.ones(3), np.random.randn(3,3)),
                                   (False, False, True), 2, torch.float64, (0,2))
        self._assert_torch_numeric(atoms.max(None))
        self._assert_torch_numeric(atoms.min(None))
        self._assert_torch_numeric(atoms.norm_inf(None))
        self._assert_torch_numeric(atoms.normNuc(None), (False, True, True))
        self._assert_torch_numeric(atoms.norm1(None))
        self._assert_torch_numeric(atoms.one_minus_pos(np.eye(3)), (True, False, True))
        self._assert_torch_numeric(atoms.pf_eigenvalue(np.eye(3)), (False, False, True))
        self._assert_torch_numeric(atoms.Pnorm(None))
        self._assert_torch_numeric(atoms.Prod(None))
        self._assert_torch_numeric(atoms.QuadForm(np.ones(3), np.eye(3)), (True, False, True),
                                   repetitions=2)
        self._assert_torch_numeric(atoms.quad_over_lin(None, None), repetitions=2)
        self._assert_torch_numeric(atoms.sigma_max(None))
        self._assert_torch_numeric(atoms.sign(None))
        self._assert_torch_numeric(atoms.tr_inv(np.eye(3)), (False, False, True))

    def test_affine_atoms(self):
        self._assert_torch_numeric(affine.binary_operators.MulExpression(np.eye(3), np.eye(3)),
                                   (True, False, True), 2, torch.float64, (1,0), (1,2))
        self._assert_torch_numeric(affine.binary_operators.multiply(np.eye(3), np.eye(3)),
                                   repetitions=2)
        self._assert_torch_numeric(affine.binary_operators.DivExpression(np.eye(3), np.eye(3)),
                                   repetitions=2)
        self._assert_torch_numeric(affine.conj.conj(None))
        self._assert_torch_numeric(affine.cumsum.cumsum(None))
        self._assert_torch_numeric(affine.diag.diag(None))
        self._assert_torch_numeric(affine.hstack.Hstack(np.ones(3), np.ones(3)), repetitions=2)
        self._assert_torch_numeric(affine.imag.imag(None), (True, True, True), 1, torch.cfloat,
                                   (3, ))
        self._assert_torch_numeric(affine.kron.kron(np.eye(3), np.eye(3)), repetitions=2)
        self._assert_torch_numeric(affine.promote.Promote(None, (3,)))
        self._assert_torch_numeric(affine.real.real(None), (True, True, True), 1, torch.cfloat,
                                   (3, ))
        self._assert_torch_numeric(affine.reshape.reshape(np.ones(3), (3,)), (True, False, False))
        self._assert_torch_numeric(affine.sum.Sum(None))
        self._assert_torch_numeric(affine.trace.trace(np.eye(3)), (False, False, True))
        self._assert_torch_numeric(affine.transpose.transpose(None))
        self._assert_torch_numeric(affine.upper_tri.upper_tri(np.eye(3)), (False, False, True))
        self._assert_torch_numeric(affine.vstack.vstack(np.ones(3)))

    def test_elementwise_atoms(self):
        self._assert_torch_numeric(elementwise.abs.abs(None))
        self._assert_torch_numeric(elementwise.ceil.ceil(None))
        self._assert_torch_numeric(elementwise.exp.exp(None))
        self._assert_torch_numeric(elementwise.log.log(None))
        self._assert_torch_numeric(elementwise.logistic.logistic(None))
        self._assert_torch_numeric(elementwise.power.power(None, 3))
        self._assert_torch_numeric(elementwise.xexp.xexp(None))
        self._assert_torch_numeric(elementwise.maximum.maximum(None, None))
        self._assert_torch_numeric(elementwise.minimum.minimum(None, None))