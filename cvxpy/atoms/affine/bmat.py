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

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.expressions.constants import Constant


def bmat(block_lists):
    """Constructs a block matrix.

    Takes a list of lists. Each internal list is stacked horizontally.
    The internal lists are stacked vertically.

    Parameters
    ----------
    block_lists : list of lists
        The blocks of the block matrix.

    Return
    ------
    CVXPY expression
        The CVXPY expression representing the block matrix.
    """
    block_lists = [
        [AffAtom.cast_to_const(block) for block in block_list]
        for block_list in block_lists
    ]
    if all(block.is_constant() for block_list in block_lists for block in block_list):
        return Constant(np.block([
            [block.value for block in block_list]
            for block_list in block_lists
        ]))

    row_blocks = [hstack(blocks) for blocks in block_lists]
    return vstack(row_blocks)
