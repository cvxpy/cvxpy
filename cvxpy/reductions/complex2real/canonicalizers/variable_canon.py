"""
Copyright 2013 Steven Diamond, 2022 - the CVXPY Authors.

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


from cvxpy.atoms.affine.upper_tri import vec_to_upper_tri
from cvxpy.atoms.affine.wraps import skew_symmetric_wrap
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variable import Variable


def _split_complex_attributes(attributes: dict) -> tuple[dict, dict]:
    """Map attributes for full-size real/imaginary split Variables.

    This helper is used for general complex and purely imaginary variables.
    Hermitian variables handle their imaginary component separately with a
    compact skew-symmetric parameterization.
    """
    real_attr = attributes.copy()
    imag_attr = attributes.copy()

    for attr in ["complex", "imag", "hermitian"]:
        real_attr.pop(attr, None)
        imag_attr.pop(attr, None)

    # PSD/NSD are meaningful for the real split variable, but not for the
    # imaginary split variable.
    imag_attr.pop("PSD", None)
    imag_attr.pop("NSD", None)

    if attributes.get("hermitian") and not (
        attributes.get("PSD") or attributes.get("NSD")
    ):
        real_attr["symmetric"] = True

    return real_attr, imag_attr


def variable_canon(expr, real_args, imag_args, real2imag):
    if expr.is_real():
        # Purely real.
        return expr, None

    elif expr.is_imag():
        # Purely imaginary.
        _, imag_attr = _split_complex_attributes(expr.attributes)
        imag = Variable(expr.shape, var_id=real2imag[expr.id], **imag_attr)
        return None, imag

    elif expr.is_complex() and expr.is_hermitian():
        n = expr.shape[0]
        real_attr, _ = _split_complex_attributes(expr.attributes)
        real = Variable((n, n), var_id=expr.id, **real_attr)

        if n > 1:
            # The imaginary part of a Hermitian matrix is skew-symmetric.
            # It is represented by a compact vector of strict upper-triangular
            # entries, not by a full matrix Variable, so matrix-shaped imag_attr
            # is intentionally not forwarded here.
            imag_var = Variable(
                shape=n * (n - 1) // 2,
                var_id=real2imag[expr.id],
            )
            imag_upper_tri = vec_to_upper_tri(imag_var, strict=True)
            imag = skew_symmetric_wrap(imag_upper_tri - imag_upper_tri.T)
        else:
            imag = Constant([[0.0]])

        return real, imag

    else:
        real_attr, imag_attr = _split_complex_attributes(expr.attributes)

        real_var = Variable(shape=expr.shape, var_id=expr.id, **real_attr)
        imag_var = Variable(shape=expr.shape, var_id=real2imag[expr.id], **imag_attr)

        return real_var, imag_var
