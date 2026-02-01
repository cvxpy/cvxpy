#ifndef ATOM_RESHAPE_H
#define ATOM_RESHAPE_H

#include "common.h"

static PyObject *py_make_reshape(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    int d1, d2;

    if (!PyArg_ParseTuple(args, "Oii", &child_capsule, &d1, &d2))
    {
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        return NULL;
    }

    expr *node = new_reshape(child, d1, d2);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create reshape node");
        return NULL;
    }

    expr_retain(node);
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_RESHAPE_H */
