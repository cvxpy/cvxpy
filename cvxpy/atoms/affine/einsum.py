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
from numpy._core.einsumfunc import (
    _find_contraction,
    _greedy_path,
    _optimal_path,
    _parse_einsum_input,
)

from cvxpy.atoms.affine.binary_operators import (
    multiply,
    reshape,
)
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.atoms.affine.transpose import permute_dims


def einsum(subscripts, *exprs, optimize=False):
    """Implement einsum using existing CVXPY atoms."""
    # Initial parsing
    dummy_operands = [np.zeros(expr.shape) for expr in exprs]
    input_subscripts, output_subscript, _ = _parse_einsum_input((
        subscripts, *dummy_operands
    ))

    input_list = input_subscripts.split(',')
    output_set = set(output_subscript)
    dimension_dict = _validate_arguments(input_list, exprs)

    # Reduce duplicate indices
    operands = []
    reduced_inputs = []
    input_sets = []
    for expr, input in zip(exprs, input_list):
        operand, inputs = _initial_reduction(expr, input, dimension_dict)
        operands.append(operand)
        reduced_inputs.append(inputs)
        input_sets.append(set(inputs))

    # If only one input, simply perform an axis sum
    if len(operands) == 1:
        return _sum_single_operand(operands[0], reduced_inputs[0], output_subscript)

    output_set = set(output_subscript)
    path = _get_path(input_sets, output_set, dimension_dict, optimize)
    
    # Contract tensors
    for contraction_inds in path:
        # Results of contraction go to the last of the list
        _, input_sets, to_remove, all_labels = _find_contraction(
            contraction_inds, 
            input_sets, 
            output_set
        )

        new_operand, new_input = _contract_pair(
            [operands[i] for i in contraction_inds],
            [reduced_inputs[i] for i in contraction_inds],
            to_remove,
            all_labels,
            dimension_dict
        )

        operands = [operands[i] for i in range(len(operands)) if i not in contraction_inds]
        operands.append(new_operand)
        reduced_inputs = [
            reduced_inputs[i] for i in range(len(reduced_inputs)) 
            if i not in contraction_inds
        ]
        reduced_inputs.append(new_input)

    # After all contractions, permute the final result to match output subscript order
    final_operand = operands[0]
    final_input = reduced_inputs[0]
    
    # Create permutation to match output subscript order
    if final_input != output_subscript:
        perm = [final_input.index(x) for x in output_subscript]
        final_operand = permute_dims(final_operand, axes=perm)

    return final_operand

def _validate_arguments(input_list, exprs):
    dimension_dict = {}

    if len(exprs) != len(input_list):
        raise ValueError(f"Number of arguments ({len(exprs)}) doesn't match "
                           f"number of input patterns ({len(input_list)})")
                        
    for tnum, term in enumerate(input_list):
        sh = exprs[tnum].shape
        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s does not contain the "
                                "correct number of indices for operand %d."
                                % (input_list[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if char in dimension_dict:
                if dimension_dict[char] != dim:
                    raise ValueError("Size of label '%s' for operand %d (%d) "
                                        "does not match previous terms (%d)."
                                        % (char, tnum, dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim

    return dimension_dict

def _get_path(input_sets, output_set, dimension_dict, optimize):
    """Get the path for the einsum operation."""
    if optimize in {True, "optimal"}:
        return _optimal_path(input_sets, output_set, dimension_dict, np.iinfo(np.int32).max)
    elif optimize in {False, "greedy"}:
        return _greedy_path(input_sets, output_set, dimension_dict, np.iinfo(np.int32).max)
    else:
        raise ValueError("Invalid value for optimize. Must be True, False, 'optimal', or 'greedy'.")

def _initial_reduction(operand, inputs, dimension_dict):
    """Reduce operands with repeated indices."""
    # Find repeated indices
    counts = {}
    for x in inputs:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1

    # If there are no repeated indices, return the operator and inputs
    repeats = [x for x, ct in counts.items() if ct > 1]
    if len(repeats) == 0:
        return operand, inputs

    # For each repeated index, get the diagonal reduction
    for x in repeats:
        to_reduce = [i for i, y in enumerate(inputs) if y == x]
        to_keep = [i for i in range(len(inputs)) if i not in to_reduce]
        perm = to_reduce + to_keep 

        idxs = np.diag_indices(n=dimension_dict[x], ndim=len(to_reduce))

        inputs = x + "".join([inputs[i] for i in to_keep])

        operand = permute_dims(operand, axes=perm)
        operand = operand[idxs]

    # Return the reduced operand and inputs
    return operand, inputs

def _sum_single_operand(operand, input_subscript, output_subscript):
    """Sum a single operand."""
    
    if len(output_subscript) < len(input_subscript):
        idxs = [i for i, x in enumerate(input_subscript) if x not in output_subscript]
        operand = cvxpy_sum(operand, axis=tuple(idxs), keepdims=False)
    elif len(output_subscript) == 0:
        operand = cvxpy_sum(operand, axis=None, keepdims=False)

    remaining = [x for x in input_subscript if x in output_subscript]
    perm = [remaining.index(x) for x in output_subscript]
    operand = permute_dims(operand, axes=perm)
    return operand

def _contract_pair(operands, input_lists, to_remove, all_labels, dimension_dict):
    """Contract a pair of operands."""
    # Align and permute the operands to compatible shapes
    aligned_operands = []
    for operand, input_list in zip(operands, input_lists):

        extra_dims = tuple(1 for x in all_labels if x not in input_list)
        if len(extra_dims) > 0:
            shape = operand.shape + extra_dims
            input_list += "".join([x for x in all_labels if x not in input_list])
            perm = [input_list.index(x) for x in all_labels]
            operand = reshape(operand, shape=shape, order="C")
        
        perm = [input_list.index(x) for x in all_labels]
        if perm != list(range(len(perm))):
            operand = permute_dims(operand, axes=perm)
        
        aligned_operands.append(operand)

    # Elementwise multiply the operands
    new_operand = multiply(aligned_operands[0], aligned_operands[1])

    # Sum the operands along the axes to be removed
    if len(to_remove) == len(all_labels):
        new_operand = cvxpy_sum(new_operand, axis=None, keepdims=False)
    elif len(to_remove) > 0:
        axes_to_remove = [i for i, x in enumerate(all_labels) if x in to_remove]
        new_operand = cvxpy_sum(new_operand, axis=tuple(axes_to_remove), keepdims=False)
    
    new_input = "".join([x for x in all_labels if x not in to_remove])

    # Return the contracted operand and inputs
    return new_operand, new_input
        