#ifndef ATOM_PROMOTE_H
#define ATOM_PROMOTE_H

#include "common.h"

static PyObject *py_make_promote(PyObject *self, PyObject *args)
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
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_promote(child, d1, d2);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create promote node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_PROMOTE_H */
