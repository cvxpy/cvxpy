#ifndef ATOM_CONSTANT_H
#define ATOM_CONSTANT_H

#include "common.h"

static PyObject *py_make_constant(PyObject *self, PyObject *args)
{
    int d1, d2, n_vars;
    PyObject *values_obj;
    if (!PyArg_ParseTuple(args, "iiiO", &d1, &d2, &n_vars, &values_obj))
    {
        return NULL;
    }

    PyArrayObject *values_array = (PyArrayObject *) PyArray_FROM_OTF(
        values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!values_array)
    {
        return NULL;
    }

    expr *node =
        new_constant(d1, d2, n_vars, (const double *) PyArray_DATA(values_array));
    Py_DECREF(values_array);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create constant node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_CONSTANT_H */
