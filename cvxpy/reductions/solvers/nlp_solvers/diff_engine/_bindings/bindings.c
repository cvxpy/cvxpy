#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

/* Include atom bindings */
#include "atoms/add.h"
#include "atoms/asinh.h"
#include "atoms/atanh.h"
#include "atoms/broadcast.h"
#include "atoms/const_scalar_mult.h"
#include "atoms/const_vector_mult.h"
#include "atoms/constant.h"
#include "atoms/cos.h"
#include "atoms/diag_vec.h"
#include "atoms/entr.h"
#include "atoms/exp.h"
#include "atoms/getters.h"
#include "atoms/hstack.h"
#include "atoms/index.h"
#include "atoms/left_matmul.h"
#include "atoms/linear.h"
#include "atoms/log.h"
#include "atoms/logistic.h"
#include "atoms/matmul.h"
#include "atoms/multiply.h"
#include "atoms/neg.h"
#include "atoms/power.h"
#include "atoms/prod.h"
#include "atoms/prod_axis_one.h"
#include "atoms/prod_axis_zero.h"
#include "atoms/promote.h"
#include "atoms/quad_form.h"
#include "atoms/quad_over_lin.h"
#include "atoms/rel_entr.h"
#include "atoms/rel_entr_scalar_vector.h"
#include "atoms/rel_entr_vector_scalar.h"
#include "atoms/reshape.h"
#include "atoms/right_matmul.h"
#include "atoms/sin.h"
#include "atoms/sinh.h"
#include "atoms/sum.h"
#include "atoms/tan.h"
#include "atoms/tanh.h"
#include "atoms/trace.h"
#include "atoms/transpose.h"
#include "atoms/variable.h"
#include "atoms/xexp.h"

/* Include problem bindings */
#include "problem/constraint_forward.h"
#include "problem/gradient.h"
#include "problem/hessian.h"
#include "problem/init_derivatives.h"
#include "problem/init_hessian.h"
#include "problem/init_jacobian.h"
#include "problem/jacobian.h"
#include "problem/make_problem.h"
#include "problem/objective_forward.h"

static int numpy_initialized = 0;

static int ensure_numpy(void)
{
    if (numpy_initialized) return 0;
    import_array1(-1);
    numpy_initialized = 1;
    return 0;
}

static PyMethodDef DNLPMethods[] = {
    {"make_variable", py_make_variable, METH_VARARGS, "Create variable node"},
    {"make_constant", py_make_constant, METH_VARARGS, "Create constant node"},
    {"make_linear", py_make_linear, METH_VARARGS, "Create linear op node"},
    {"make_log", py_make_log, METH_VARARGS, "Create log node"},
    {"make_exp", py_make_exp, METH_VARARGS, "Create exp node"},
    {"make_index", py_make_index, METH_VARARGS, "Create index node"},
    {"make_add", py_make_add, METH_VARARGS, "Create add node"},
    {"make_trace", py_make_trace, METH_VARARGS, "Create trace node"},
    {"make_transpose", py_make_transpose, METH_VARARGS, "Create transpose node"},
    {"make_hstack", py_make_hstack, METH_VARARGS,
     "Create hstack node from list of expr capsules and n_vars (make_hstack([e1, "
     "e2, ...], n_vars))"},
    {"make_sum", py_make_sum, METH_VARARGS, "Create sum node"},
    {"make_neg", py_make_neg, METH_VARARGS, "Create neg node"},
    {"make_promote", py_make_promote, METH_VARARGS, "Create promote node"},
    {"make_multiply", py_make_multiply, METH_VARARGS,
     "Create elementwise multiply node"},
    {"make_matmul", py_make_matmul, METH_VARARGS,
     "Create matrix multiplication node (Z = X @ Y)"},
    {"make_const_scalar_mult", py_make_const_scalar_mult, METH_VARARGS,
     "Create constant scalar multiplication node (a * f(x))"},
    {"make_const_vector_mult", py_make_const_vector_mult, METH_VARARGS,
     "Create constant vector multiplication node (a ∘ f(x))"},
    {"make_power", py_make_power, METH_VARARGS, "Create power node"},
    {"make_prod", py_make_prod, METH_VARARGS, "Create prod node"},
    {"make_prod_axis_zero", py_make_prod_axis_zero, METH_VARARGS,
     "Create prod_axis_zero node"},
    {"make_prod_axis_one", py_make_prod_axis_one, METH_VARARGS,
     "Create prod_axis_one node"},
    {"make_sin", py_make_sin, METH_VARARGS, "Create sin node"},
    {"make_cos", py_make_cos, METH_VARARGS, "Create cos node"},
    {"make_diag_vec", py_make_diag_vec, METH_VARARGS, "Create diag_vec node"},
    {"make_tan", py_make_tan, METH_VARARGS, "Create tan node"},
    {"make_sinh", py_make_sinh, METH_VARARGS, "Create sinh node"},
    {"make_tanh", py_make_tanh, METH_VARARGS, "Create tanh node"},
    {"make_asinh", py_make_asinh, METH_VARARGS, "Create asinh node"},
    {"make_atanh", py_make_atanh, METH_VARARGS, "Create atanh node"},
    {"make_broadcast", py_make_broadcast, METH_VARARGS, "Create broadcast node"},
    {"make_entr", py_make_entr, METH_VARARGS, "Create entr node"},
    {"make_logistic", py_make_logistic, METH_VARARGS, "Create logistic node"},
    {"make_xexp", py_make_xexp, METH_VARARGS, "Create xexp node"},
    {"make_left_matmul", py_make_left_matmul, METH_VARARGS,
     "Create left matmul node (A @ f(x))"},
    {"make_right_matmul", py_make_right_matmul, METH_VARARGS,
     "Create right matmul node (f(x) @ A)"},
    {"make_quad_form", py_make_quad_form, METH_VARARGS,
     "Create quadratic form node (x' * Q * x)"},
    {"make_quad_over_lin", py_make_quad_over_lin, METH_VARARGS,
     "Create quad_over_lin node (sum(x^2) / y)"},
    {"make_rel_entr", py_make_rel_entr, METH_VARARGS,
     "Create rel_entr node: x * log(x/y) elementwise"},
    {"make_rel_entr_vector_scalar", py_make_rel_entr_vector_scalar, METH_VARARGS,
     "Create rel_entr node with vector first arg, scalar second arg"},
    {"make_rel_entr_scalar_vector", py_make_rel_entr_scalar_vector, METH_VARARGS,
     "Create rel_entr node with scalar first arg, vector second arg"},
    {"get_expr_dimensions", py_get_expr_dimensions, METH_VARARGS,
     "Get the dimensions (d1, d2) of an expression"},
    {"get_expr_size", py_get_expr_size, METH_VARARGS,
     "Get the total size of an expression"},
    {"make_reshape", py_make_reshape, METH_VARARGS, "Create reshape atom"},
    {"make_problem", py_make_problem, METH_VARARGS,
     "Create problem from objective and constraints"},
    {"problem_init_derivatives", py_problem_init_derivatives, METH_VARARGS,
     "Initialize derivative structures"},
    {"problem_init_jacobian", py_problem_init_jacobian, METH_VARARGS,
     "Initialize Jacobian structures only"},
    {"problem_init_hessian", py_problem_init_hessian, METH_VARARGS,
     "Initialize Hessian structures only"},
    {"problem_objective_forward", py_problem_objective_forward, METH_VARARGS,
     "Evaluate objective only"},
    {"problem_constraint_forward", py_problem_constraint_forward, METH_VARARGS,
     "Evaluate constraints only"},
    {"problem_gradient", py_problem_gradient, METH_VARARGS,
     "Compute objective gradient"},
    {"problem_jacobian", py_problem_jacobian, METH_VARARGS,
     "Compute constraint jacobian"},
    {"get_jacobian", py_get_jacobian, METH_VARARGS,
     "Get constraint jacobian without recomputing"},
    {"problem_hessian", py_problem_hessian, METH_VARARGS,
     "Compute Lagrangian Hessian"},
    {"get_hessian", py_get_hessian, METH_VARARGS,
     "Get Lagrangian Hessian without recomputing"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef dnlp_module = {PyModuleDef_HEAD_INIT, "_diffengine", NULL,
                                         -1, DNLPMethods};

PyMODINIT_FUNC PyInit__diffengine(void)
{
    if (ensure_numpy() < 0) return NULL;
    return PyModule_Create(&dnlp_module);
}
