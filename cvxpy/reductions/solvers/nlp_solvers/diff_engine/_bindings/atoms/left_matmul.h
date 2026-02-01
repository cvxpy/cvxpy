#ifndef ATOM_LEFT_MATMUL_H
#define ATOM_LEFT_MATMUL_H

#include "bivariate.h"
#include "common.h"

/* Left matrix multiplication: A @ f(x) where A is a constant matrix */
static PyObject *py_make_left_matmul(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    int m, n;
    if (!PyArg_ParseTuple(args, "OOOOii", &child_capsule, &data_obj, &indices_obj,
                          &indptr_obj, &m, &n))
    {
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    PyArrayObject *data_array =
        (PyArrayObject *) PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *indices_array = (PyArrayObject *) PyArray_FROM_OTF(
        indices_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *indptr_array = (PyArrayObject *) PyArray_FROM_OTF(
        indptr_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!data_array || !indices_array || !indptr_array)
    {
        Py_XDECREF(data_array);
        Py_XDECREF(indices_array);
        Py_XDECREF(indptr_array);
        return NULL;
    }

    int nnz = (int) PyArray_SIZE(data_array);
    CSR_Matrix *A = new_csr_matrix(m, n, nnz);
    memcpy(A->x, PyArray_DATA(data_array), nnz * sizeof(double));
    memcpy(A->i, PyArray_DATA(indices_array), nnz * sizeof(int));
    memcpy(A->p, PyArray_DATA(indptr_array), (m + 1) * sizeof(int));

    Py_DECREF(data_array);
    Py_DECREF(indices_array);
    Py_DECREF(indptr_array);

    expr *node = new_left_matmul(child, A);
    free_csr_matrix(A);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create left_matmul node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_LEFT_MATMUL_H */
