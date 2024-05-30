use crate::backend::CONST_ID;
use crate::tensor_representation::TensorRepresentation;
use crate::{
    faer_ext::{self, to_triplets_iter},
    IdxMap,
};
use faer::sparse::SparseColMat;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Default, Debug)]
pub(crate) struct ViewContext {
    /// Maps variable id to first column associated with its entries
    pub(crate) id_to_col: IdxMap, 
    /// Maps parameter id to number of entries in parameter
    pub(crate) param_to_size: IdxMap, 
    /// Maps parameter id to first matrix/slice (column in a 3D
    /// sense) associated with its entries
    pub(crate) param_to_col: IdxMap,
    /// Total number of parameter entries + 1
    pub(crate) param_size_plus_one: i64,
    /// Total number of variables in problem
    pub(crate) var_length: i64, 
}
type VarId = i64;
type ParamId = i64;

pub(crate) type Tensor = HashMap<VarId, HashMap<ParamId, crate::SparseMatrix>>;

#[derive(Clone, Debug)]
pub(crate) struct View<'a> {
    pub(crate) variables: HashSet<i64>,
    pub(crate) tensor: Tensor,
    pub(crate) is_parameter_free: bool,
    pub(crate) context: &'a ViewContext,
}

impl<'a> View<'a> {
    pub fn new(context: &'a ViewContext) -> Self {
        View {
            variables: HashSet::new(),
            tensor: Tensor::new(),
            is_parameter_free: true,
            context,
        }
    }

    pub fn get_tensor_representation(&self, row_offset: i64) -> TensorRepresentation {
        let mut tensor_representations = Vec::new();

        for (&variable_id, variable_tensor) in self.tensor.iter() {
            for (&parameter_id, parameter_matrix) in variable_tensor.iter() {
                let p = self.context.param_to_size[&parameter_id];
                let m = parameter_matrix.nrows() as i64 / p;
                let (new_rows, new_cols, data, new_param_offset): (
                    Vec<u64>,
                    Vec<u64>,
                    Vec<f64>,
                    Vec<u64>,
                ) = to_triplets_iter(parameter_matrix)
                    .map(|(i, j, d)| {
                        let row_index = (i as i64 % m + row_offset) as u64;
                        let col_index = (j as i64 + self.context.id_to_col[&variable_id]) as u64;
                        let param_offset =
                            (i as i64 / m + self.context.param_to_col[&parameter_id]) as u64;
                        (row_index, col_index, d, param_offset)
                    })
                    .fold(
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                        |(mut rows, mut cols, mut data, mut param_offset), (row, col, d, param)| {
                            rows.push(row);
                            cols.push(col);
                            data.push(d);
                            param_offset.push(param);
                            (rows, cols, data, param_offset)
                        },
                    );

                // Add to tensor_representations
                tensor_representations.push(TensorRepresentation {
                    data,
                    row: new_rows,
                    col: new_cols,
                    parameter_offset: new_param_offset,
                });
            }
        }

        TensorRepresentation::combine(tensor_representations)
    }

    pub fn apply_all<F>(&mut self, mut func: F)
    where
        F: FnMut(&crate::SparseMatrix, i64) -> crate::SparseMatrix,
    {
        self.tensor = self
            .tensor
            .iter()
            .map(|(var_id, parameter_repr)| {
                (
                    *var_id,
                    parameter_repr
                        .iter()
                        .map(|(k, v)| (*k, func(v, self.context.param_to_size[k])))
                        .collect(),
                )
            })
            .collect();
    }

    pub(crate) fn select_rows(&mut self, rows: &[u64]) {
        let func = |x: &SparseColMat<u64, f64>, p: i64| -> crate::SparseMatrix {
            if p == 1 {
                faer_ext::select_rows(x, rows)
            } else {
                let m = (x.nrows() / p as usize) as u64;
                let mut new_rows = Vec::with_capacity(rows.len() * p as usize);
                for i in 0..p as u64 {
                    for &r in rows {
                        new_rows.push(r + m * i);
                    }
                }
                faer_ext::select_rows(x, rows)
            }
        };

        self.apply_all(func);
    }

    pub(crate) fn rows(&self) -> u64 {
        for (_, tensor) in &self.tensor {
            for (param_id, param_mat) in tensor {
                return param_mat.nrows() as u64 / self.context.param_to_size[&param_id] as u64;
            }
            panic!("No parameters in tensor");
        }
        panic!("No variables in tensor");
    }

    pub(crate) fn accumulate_over_variables(
        mut self,
        func: &impl Fn(&SparseColMat<u64, f64>, u64) -> SparseColMat<u64, f64>,
        is_parameter_free_function: bool,
    ) -> Self {
        for (variable_id, tensor) in self.tensor.iter_mut() {

            *tensor = if is_parameter_free_function {
                View::apply_to_parameters(func, tensor)
            } else {
                // func(&tensor[&CONST_ID], 1)
                todo!("Implement accumulate_over_variables")
            };
        }

        let is_parameter_free = self.is_parameter_free && is_parameter_free_function;
        self
    }

    pub(crate) fn apply_to_parameters(
        func: impl Fn(&SparseColMat<u64, f64>, u64) -> SparseColMat<u64, f64>,
        tensor: &HashMap<i64, SparseColMat<u64, f64>>,
    ) -> HashMap<i64, SparseColMat<u64, f64>> {
        tensor
            .iter()
            .map(|(k, v)| (*k, func(v, 1)))
            .collect()
    }
    
    pub(crate) fn add_inplace(&self, arg_res: &View) -> () {
        self.variables.extend(&arg_res.variables);
        self.is_parameter_free = self.is_parameter_free && arg_res.is_parameter_free;
        extend_tensor_outer(&mut self.tensor, &arg_res.tensor);
    }
}

fn extend_tensor_outer(tensor_1: &mut Tensor, tensor_2: &Tensor) -> () {

    let keys_a: HashSet<i64> = tensor_1.keys().cloned().collect();
    let keys_b: HashSet<i64> = tensor_2.keys().cloned().collect();
    let intersect: HashSet<i64> = keys_a.intersection(&keys_b).cloned().collect();
    let union: HashSet<i64> = keys_a.union(&keys_b).cloned().collect();

    for key in union {
        if intersect.contains(&key) {
            let mut tensor_1_val = tensor_1.get(&key).unwrap();
            let tensor_2_val = tensor_2.get(&key).unwrap();
            extend_tensor_inner(&mut tensor_1_val, tensor_2_val);
        } else if keys_a.contains(&key) {
            let tensor_1_val = tensor_1.get(&key).unwrap();
            tensor_1.insert(key, tensor_1_val.clone());
        } else {
            let tensor_2_val = tensor_2.get(&key).unwrap();
            tensor_1.insert(key, tensor_2_val.clone());
        }
    }
}

fn extend_tensor_inner(tensor_1: &mut HashMap<i64, SparseColMat<u64, f64>>, tensor_2: &HashMap<i64, SparseColMat<u64, f64>>) -> () {
    let keys_a: HashSet<i64> = tensor_1.keys().cloned().collect();
    let keys_b: HashSet<i64> = tensor_2.keys().cloned().collect();
    let intersect: HashSet<i64> = keys_a.intersection(&keys_b).cloned().collect();
    let union: HashSet<i64> = keys_a.union(&keys_b).cloned().collect();

    for key in union {
        if intersect.contains(&key) {
            let mut tensor_1_val = tensor_1.get(&key).unwrap();
            let tensor_2_val = tensor_2.get(&key).unwrap();
            let new_tensor = tensor_1_val + tensor_2_val;
            tensor_1.insert(key, new_tensor);
        } else if keys_a.contains(&key) {
            let tensor_1_val = tensor_1.get(&key).unwrap();
            tensor_1.insert(key, tensor_1_val.clone());
        } else {
            let tensor_2_val = tensor_2.get(&key).unwrap();
            tensor_1.insert(key, tensor_2_val.clone());
        }
    }
}