use pyo3::prelude::*;
use crate::IdxMap;
use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct ViewContext {
    pub(crate) id_to_col: IdxMap,
    pub(crate) param_to_size: IdxMap,
    pub(crate) param_to_col: IdxMap,
    pub(crate) param_size_plus_one: i64,
    pub(crate) var_length: i64,
}
type VarId = i64;
type ParamId = i64;

pub(crate) type Tensor = HashMap<VarId, HashMap<ParamId, crate::SparseMatrix>>;

pub(crate) struct View<'a> {
    pub(crate) variables: Vec<i64>,
    pub(crate) tensor: Tensor,
    pub(crate) is_parameter_free: bool,
    pub(crate) context: &'a ViewContext,
}

impl<'a> View<'a> {
    pub fn new(context: &'a ViewContext) -> Self {
        return View {
            variables: Vec::new(),
            tensor: Tensor::new(),
            is_parameter_free: false,
            context,
        }
    }

    pub fn from_tensor(t: Tensor) -> Self {
    }
}
