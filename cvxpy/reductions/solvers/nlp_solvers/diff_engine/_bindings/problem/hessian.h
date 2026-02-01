#ifndef PROBLEM_HESSIAN_H
#define PROBLEM_HESSIAN_H

#include "common.h"

/*
 * py_problem_hessian: Compute Lagrangian Hessian
 *
 * Args:
 *   prob_capsule: PyCapsule containing problem pointer
 *   obj_factor: Scaling factor for objective Hessian (double)
 *   lagrange: Array of Lagrange multipliers (numpy array, length =
 * total_constraint_size)
 *
 * Returns:
 *   Tuple of (data, indices, indptr, (m, n)) for scipy.sparse.csr_matrix
 */
static PyObject *py_problem_hessian(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    double obj_factor;
    PyObject *lagrange_obj;

    if (!PyArg_ParseTuple(args, "OdO", &prob_capsule, &obj_factor, &lagrange_obj))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    /* Convert lagrange to contiguous C array */
    PyArrayObject *lagrange_arr = (PyArrayObject *) PyArray_FROM_OTF(
        lagrange_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!lagrange_arr)
    {
        return NULL;
    }

    double *lagrange = (double *) PyArray_DATA(lagrange_arr);

    /* Compute Hessian */
    problem_hessian(prob, obj_factor, lagrange);

    Py_DECREF(lagrange_arr);

    /* Extract CSR components and return as tuple */
    CSR_Matrix *H = prob->lagrange_hessian;
    npy_intp nnz = H->nnz;
    npy_intp n_plus_1 = H->n + 1;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    PyObject *indices = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *indptr = PyArray_SimpleNew(1, &n_plus_1, NPY_INT32);

    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        return NULL;
    }

    /* Copy CSR data using memcpy for efficiency */
    memcpy(PyArray_DATA((PyArrayObject *) data), H->x, nnz * sizeof(double));
    memcpy(PyArray_DATA((PyArrayObject *) indices), H->i, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) indptr), H->p, n_plus_1 * sizeof(int));

    return Py_BuildValue("(OOO(ii))", data, indices, indptr, H->m, H->n);
}

static PyObject *py_get_hessian(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    if (!PyArg_ParseTuple(args, "O", &prob_capsule))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    if (!prob->lagrange_hessian)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "hessian not initialized - call problem_hessian first");
        return NULL;
    }

    CSR_Matrix *H = prob->lagrange_hessian;
    npy_intp nnz = H->nnz;
    npy_intp n_plus_1 = H->n + 1;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    PyObject *indices = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *indptr = PyArray_SimpleNew(1, &n_plus_1, NPY_INT32);

    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) data), H->x, nnz * sizeof(double));
    memcpy(PyArray_DATA((PyArrayObject *) indices), H->i, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) indptr), H->p, n_plus_1 * sizeof(int));

    return Py_BuildValue("(OOO(ii))", data, indices, indptr, H->m, H->n);
}

#endif /* PROBLEM_HESSIAN_H */
