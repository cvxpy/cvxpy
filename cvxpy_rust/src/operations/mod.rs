//! LinOp operation implementations
//!
//! This module contains implementations for all 22 LinOp operation types.
//! Each operation transforms input tensors according to the operation semantics.

mod leaf;
mod arithmetic;
mod structural;
mod specialized;

pub use leaf::*;
pub use arithmetic::*;
pub use structural::*;
pub use specialized::*;

use crate::linop::{LinOp, OpType};
use crate::tensor::SparseTensor;
use std::collections::HashMap;

/// Context for processing LinOp trees
#[derive(Clone)]
pub struct ProcessingContext {
    pub id_to_col: HashMap<i64, i64>,
    pub param_to_size: HashMap<i64, i64>,
    pub param_to_col: HashMap<i64, i64>,
    pub var_length: i64,
    pub param_size_plus_one: i64,
}

impl ProcessingContext {
    /// Get the column offset for a variable ID
    pub fn var_col(&self, var_id: i64) -> i64 {
        *self.id_to_col.get(&var_id).unwrap_or(&0)
    }

    /// Get the column offset for a parameter ID
    pub fn param_col(&self, param_id: i64) -> i64 {
        *self.param_to_col.get(&param_id).unwrap_or(&0)
    }

    /// Get the size of a parameter
    pub fn param_size(&self, param_id: i64) -> i64 {
        *self.param_to_size.get(&param_id).unwrap_or(&1)
    }

    /// Get the constant column offset (var_length for the 'b' column)
    pub fn const_col(&self) -> i64 {
        self.var_length
    }

    /// Get the constant parameter offset (last slice for non-parametric)
    pub fn const_param(&self) -> i64 {
        self.param_size_plus_one - 1
    }
}

/// Process a LinOp node and its children recursively
///
/// This is the main entry point for converting a LinOp tree to a SparseTensor.
pub fn process_linop(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    match lin_op.op_type {
        // Leaf nodes
        OpType::Variable => leaf::process_variable(lin_op, ctx),
        OpType::ScalarConst => leaf::process_scalar_const(lin_op, ctx),
        OpType::DenseConst => leaf::process_dense_const(lin_op, ctx),
        OpType::SparseConst => leaf::process_sparse_const(lin_op, ctx),
        OpType::Param => leaf::process_param(lin_op, ctx),

        // Trivial operations
        OpType::Sum => process_sum(lin_op, ctx),
        OpType::Neg => arithmetic::process_neg(lin_op, ctx),
        OpType::Reshape => process_reshape(lin_op, ctx),

        // Arithmetic operations
        OpType::Mul => arithmetic::process_mul(lin_op, ctx),
        OpType::Rmul => arithmetic::process_rmul(lin_op, ctx),
        OpType::MulElem => arithmetic::process_mul_elem(lin_op, ctx),
        OpType::Div => arithmetic::process_div(lin_op, ctx),

        // Structural operations
        OpType::Index => structural::process_index(lin_op, ctx),
        OpType::Transpose => structural::process_transpose(lin_op, ctx),
        OpType::Promote => structural::process_promote(lin_op, ctx),
        OpType::BroadcastTo => structural::process_broadcast_to(lin_op, ctx),
        OpType::Hstack => structural::process_hstack(lin_op, ctx),
        OpType::Vstack => structural::process_vstack(lin_op, ctx),
        OpType::Concatenate => structural::process_concatenate(lin_op, ctx),

        // Specialized operations
        OpType::SumEntries => specialized::process_sum_entries(lin_op, ctx),
        OpType::Trace => specialized::process_trace(lin_op, ctx),
        OpType::DiagVec => specialized::process_diag_vec(lin_op, ctx),
        OpType::DiagMat => specialized::process_diag_mat(lin_op, ctx),
        OpType::UpperTri => specialized::process_upper_tri(lin_op, ctx),
        OpType::Conv => specialized::process_conv(lin_op, ctx),
        OpType::KronR => specialized::process_kron_r(lin_op, ctx),
        OpType::KronL => specialized::process_kron_l(lin_op, ctx),

        // No-op
        OpType::NoOp => SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1)),
    }
}

/// Sum operation - accumulates results from all args (NOOP for single arg)
fn process_sum(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Process all arguments and combine
    let mut result = process_linop(&lin_op.args[0], ctx);
    for arg in &lin_op.args[1..] {
        let arg_tensor = process_linop(arg, ctx);
        result.extend(arg_tensor);
    }
    result
}

/// Reshape operation - just passes through since we use COO format
fn process_reshape(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Reshape is a NOOP in COO format - the row indices already encode position
    process_linop(&lin_op.args[0], ctx)
}
