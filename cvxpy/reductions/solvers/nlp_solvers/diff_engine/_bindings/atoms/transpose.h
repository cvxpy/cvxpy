#ifndef PYTHON_ATOMS_TRANSPOSE_H
#define PYTHON_ATOMS_TRANSPOSE_H

#include "common.h"

// Python binding for the transpose atom
static PyObject *py_make_transpose(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    if (!PyArg_ParseTuple(args, "O", &child_capsule))
    {
        return NULL;
    }
    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);

    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_transpose(child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create trace node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif // PYTHON_ATOMS_TRANSPOSE_H
