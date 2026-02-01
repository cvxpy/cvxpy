#ifndef ATOM_SUM_H
#define ATOM_SUM_H

#include "common.h"

static PyObject *py_make_sum(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    int axis;
    if (!PyArg_ParseTuple(args, "Oi", &child_capsule, &axis))
    {
        return NULL;
    }
    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_sum(child, axis);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create sum node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_SUM_H */
