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

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.expressions.expression import Expression


def bmat(block_lists) -> Expression:
    """Constructs a block matrix.

    Takes a list of lists. Each internal list is stacked horizontally.
    The internal lists are stacked vertically.

    Scalars and 1-D blocks are promoted to 2-D so they can be combined
    with matrices, mirroring the behavior of ``numpy.block`` (e.g. when
    building an LMI that mixes scalars, vectors, and matrices).

    Parameters
    ----------
    block_lists : list of lists
        The blocks of the block matrix.

    Return
    ------
    CVXPY expression
        The CVXPY expression representing the block matrix.
    """
    row_blocks = [hstack([_promote_to_2d(block) for block in blocks])
                  for blocks in block_lists]
    return vstack(row_blocks)


def _promote_to_2d(block) -> Expression:
    """Promote a scalar or 1-D block to a 2-D row, like ``numpy.block``."""
    block = Expression.cast_to_const(block)
    if block.ndim == 0:
        return reshape(block, (1, 1), order='F')
    if block.ndim == 1:
        return reshape(block, (1, block.shape[0]), order='F')
    return block
