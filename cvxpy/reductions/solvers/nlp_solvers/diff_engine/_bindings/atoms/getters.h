#ifndef ATOM_GETTERS_H
#define ATOM_GETTERS_H

#include "common.h"

static PyObject *py_get_expr_dimensions(PyObject *self, PyObject *args)
{
    PyObject *expr_capsule;

    if (!PyArg_ParseTuple(args, "O", &expr_capsule))
    {
        return NULL;
    }

    expr *node = (expr *) PyCapsule_GetPointer(expr_capsule, EXPR_CAPSULE_NAME);
    if (!node)
    {
        return NULL;
    }

    // Return tuple (d1, d2)
    return Py_BuildValue("(ii)", node->d1, node->d2);
}

static PyObject *py_get_expr_size(PyObject *self, PyObject *args)
{
    PyObject *expr_capsule;

    if (!PyArg_ParseTuple(args, "O", &expr_capsule))
    {
        return NULL;
    }

    expr *node = (expr *) PyCapsule_GetPointer(expr_capsule, EXPR_CAPSULE_NAME);
    if (!node)
    {
        return NULL;
    }

    return Py_BuildValue("i", node->size);
}

#endif /* ATOM_GETTERS_H */
