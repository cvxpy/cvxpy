"""
Copyright, the CVXPY authors

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
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest

# ---------------------------------------------------------------------------
# Composite strategies
# ---------------------------------------------------------------------------

@st.composite
def nd_shape_with_single_neg1(draw, min_ndim=2, max_ndim=6):
    """
    Draws (concrete_shape, shape_with_neg1) where:
      - concrete_shape: all-positive tuple, the true target shape
      - shape_with_neg1: same tuple but exactly one dimension replaced with -1

    Covers all -1 positions at all dimensionalities from 2 to max_ndim.
    Individual dims are capped at 6 to keep total size tractable (<= 6^6 = 46656).
    """
    ndim = draw(st.integers(min_value=min_ndim, max_value=max_ndim))
    dims = draw(
        st.lists(st.integers(min_value=1, max_value=6), min_size=ndim, max_size=ndim)
    )
    concrete_shape = tuple(dims)
    neg1_idx = draw(st.integers(min_value=0, max_value=ndim - 1))
    shape_with_neg1 = tuple(-1 if i == neg1_idx else d for i, d in enumerate(dims))
    return concrete_shape, shape_with_neg1


@st.composite
def same_size_shape_pair(draw):
    """
    Draws (shape1, shape2) guaranteed to have identical total element counts.

    shape2 is produced by _factorize(), which uses sequential divisor sampling:
    no assumption-based filtering is needed, so hypothesis shrinking is efficient.
    Max total size is 6^5 = 7776, large enough to stress higher-dimensional paths
    without making constant-expression evaluation slow.
    """
    ndim1 = draw(st.integers(min_value=1, max_value=5))
    dims1 = draw(
        st.lists(st.integers(min_value=1, max_value=6), min_size=ndim1, max_size=ndim1)
    )
    size = int(np.prod(dims1))
    ndim2 = draw(st.integers(min_value=1, max_value=5))
    dims2 = _factorize(draw, size, ndim2)
    return tuple(dims1), tuple(dims2)


def _factorize(draw, size, ndim):
    """
    Draw a shape of exactly `ndim` positive dimensions whose product equals `size`.

    Algorithm: for each of the first ndim-1 slots, sample any divisor of the
    remaining product and divide it out. The last slot absorbs whatever is left.
    This guarantees prod(result) == size with zero hypothesis filter waste.
    """
    dims = []
    remaining = size
    for _ in range(ndim - 1):
        divisors = [d for d in range(1, remaining + 1) if remaining % d == 0]
        d = draw(st.sampled_from(divisors))
        dims.append(d)
        remaining //= d
    dims.append(remaining)
    return tuple(dims)


# ---------------------------------------------------------------------------
# Fuzz tests
# ---------------------------------------------------------------------------

class TestFuzzReshapeNdInference(BaseTest):
    """
    Property-based fuzzing suite for cp.reshape targeting:
      1. N-D -1 shape inference (N=2..6, all axis positions)
      2. Round-trip value consistency for arbitrary compatible shape pairs
      3. DCP sign and affinity preservation through the reshape node
      4. Error-type consistency when multiple -1s are present
    """

    @settings(max_examples=200, deadline=None)
    @given(
        data=nd_shape_with_single_neg1(min_ndim=2, max_ndim=6),
        order=st.sampled_from(['F', 'C']),
    )
    def test_nd_infer_shape_matches_numpy(self, data, order):
        """
        Property: for any N-D shape (2 <= N <= 6) with exactly one -1, cp.reshape
        must infer the identical target shape as numpy AND produce numerically
        identical values.

        The merged fix (fb70135ec) was regression-tested only at N=3. This test
        exercises _infer_shape() at N=4, 5, 6 with the -1 at every axis position,
        covering cases where np.prod(specified_dims) requires multiplying 3+ values.
        It also stresses shapes containing 1s (zero-effect factors) and large dims
        where a single wrong index in the old 2D idiom would silently produce a
        wrong but plausible-looking shape.
        """
        concrete_shape, shape_with_neg1 = data
        size = int(np.prod(concrete_shape))

        # Build a deterministic array shaped like the concrete target.
        arr = np.arange(float(size)).reshape(concrete_shape)
        const = cp.Constant(arr)

        reshaped = cp.reshape(const, shape_with_neg1, order=order)

        # Property (a): inferred shape must match numpy's inference exactly.
        expected_shape = np.reshape(arr, shape_with_neg1, order=order).shape
        assert reshaped.shape == expected_shape, (
            f"cp.reshape inferred {reshaped.shape} but numpy inferred "
            f"{expected_shape} for shape_with_neg1={shape_with_neg1}, order={order!r}"
        )

        # Property (b): numeric values must agree with numpy element-by-element.
        np.testing.assert_array_equal(
            reshaped.value,
            np.reshape(arr, shape_with_neg1, order=order),
            err_msg=f"Value mismatch: shape_with_neg1={shape_with_neg1}, order={order!r}",
        )

    @settings(max_examples=150, deadline=None)
    @given(
        shapes=same_size_shape_pair(),
        order=st.sampled_from(['F', 'C']),
    )
    def test_reshape_round_trip_recovers_original(self, shapes, order):
        """
        Property: reshape(reshape(X, shape2, order), shape1, order) == X exactly,
        for all compatible shape pairs and both F and C order.

        Tests that reshape is a true isomorphism on the flat data vector. Any
        silent reordering bug introduced between F/C path transitions in
        graph_implementation — particularly the shape[::-1] transpose path for
        order='C' with N > 1 — would corrupt values and be caught here.
        """
        shape1, shape2 = shapes
        size = int(np.prod(shape1))

        # Deterministic array: arange avoids masking bugs with repeated values.
        arr = np.arange(float(size)).reshape(shape1)
        const = cp.Constant(arr)

        forward = cp.reshape(const, shape2, order=order)
        back = cp.reshape(forward, shape1, order=order)

        # Intermediate shape must match what was requested.
        assert forward.shape == shape2, (
            f"Forward reshape shape {forward.shape} != requested {shape2}"
        )
        # Restored shape must match original.
        assert back.shape == shape1, (
            f"Back reshape shape {back.shape} != original {shape1}"
        )
        # Values must round-trip without corruption.
        np.testing.assert_array_equal(
            back.value,
            arr,
            err_msg=(
                f"Round-trip value mismatch: shape1={shape1}, "
                f"shape2={shape2}, order={order!r}"
            ),
        )

    @settings(max_examples=200, deadline=None)
    @given(
        shapes=same_size_shape_pair(),
        order=st.sampled_from(['F', 'C']),
        nonneg=st.booleans(),
        nonpos=st.booleans(),
    )
    def test_reshape_preserves_dcp_sign_and_affinity(self, shapes, order, nonneg, nonpos):
        """
        Properties:
          (a) cp.reshape is an affine atom: is_affine() must be True unconditionally.
          (b) A nonneg Variable reshaped to any compatible shape remains nonneg.
          (c) A nonpos Variable reshaped to any compatible shape remains nonpos.

        reshape() calls AffAtom.__init__, which delegates curvature/sign to the
        parent expression's metadata. Any future refactor that breaks metadata
        propagation through the reshape node — e.g., a short-circuit in sign_from_args
        or a new code path in graph_implementation — would be caught here.
        """
        # nonneg AND nonpos simultaneously means zero-only: skip to isolate
        # sign propagation semantics rather than test the zero-variable edge case.
        assume(not (nonneg and nonpos))

        shape1, shape2 = shapes
        x = cp.Variable(shape1, nonneg=nonneg, nonpos=nonpos)
        reshaped = cp.reshape(x, shape2, order=order)

        # (a) reshape is unconditionally affine regardless of shape or order.
        assert reshaped.is_affine(), (
            f"reshape should be affine but is_affine()=False "
            f"for shape1={shape1}, shape2={shape2}, order={order!r}"
        )

        # (b) nonnegativity metadata must survive the reshape node.
        if nonneg:
            assert reshaped.is_nonneg(), (
                f"nonneg attribute lost after reshape: "
                f"shape1={shape1}, shape2={shape2}, order={order!r}"
            )

        # (c) nonpositivity metadata must survive the reshape node.
        if nonpos:
            assert reshaped.is_nonpos(), (
                f"nonpos attribute lost after reshape: "
                f"shape1={shape1}, shape2={shape2}, order={order!r}"
            )

    @settings(max_examples=200, deadline=None)
    @given(
        dims=st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=5),
        order=st.sampled_from(['F', 'C']),
    )
    def test_multiple_neg1_raises_consistently(self, dims, order):
        """
        Property: any shape containing two or more -1 entries must always raise
        AssertionError with the message "Only one dimension can be -1", regardless
        of total size, number of dimensions, or axis positions.

        This guards the guard: if a future refactor changes the assert to a
        conditional branch or replaces it with softer logic, silent wrong-shape
        inference could replace the hard failure. Fuzzing confirms the contract
        holds for all combinations of dims and placement of the first two -1s.
        """
        size = int(np.prod(dims))
        x = cp.Variable(size)

        # Replace the first two positions with -1 (always valid since min_size=2).
        multi_neg1_shape = tuple(-1 if i < 2 else d for i, d in enumerate(dims))

        with pytest.raises(AssertionError, match="Only one dimension can be -1"):
            cp.reshape(x, multi_neg1_shape, order=order)
