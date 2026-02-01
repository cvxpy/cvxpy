// SPDX-License-Identifier: Apache-2.0

#ifndef ATOM_INDEX_H
#define ATOM_INDEX_H

#include "affine.h"
#include "common.h"

/* Index/slicing: y = child[indices] where indices is a list of flattened positions
 */
static PyObject *py_make_index(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    int d1, d2;
    PyObject *indices_obj;

    if (!PyArg_ParseTuple(args, "OiiO", &child_capsule, &d1, &d2, &indices_obj))
    {
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    /* Convert indices array to int32 */
    PyArrayObject *indices_array = (PyArrayObject *) PyArray_FROM_OTF(
        indices_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!indices_array)
    {
        return NULL;
    }

    int n_idxs = (int) PyArray_SIZE(indices_array);
    int *indices_data = (int *) PyArray_DATA(indices_array);

    expr *node = new_index(child, d1, d2, indices_data, n_idxs);

    Py_DECREF(indices_array);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create index node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_INDEX_H */
