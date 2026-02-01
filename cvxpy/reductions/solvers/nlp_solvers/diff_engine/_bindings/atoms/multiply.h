#ifndef ATOM_MULTIPLY_H
#define ATOM_MULTIPLY_H

#include "bivariate.h"
#include "common.h"

static PyObject *py_make_multiply(PyObject *self, PyObject *args)
{
    PyObject *left_capsule, *right_capsule;
    if (!PyArg_ParseTuple(args, "OO", &left_capsule, &right_capsule))
    {
        return NULL;
    }
    expr *left = (expr *) PyCapsule_GetPointer(left_capsule, EXPR_CAPSULE_NAME);
    expr *right = (expr *) PyCapsule_GetPointer(right_capsule, EXPR_CAPSULE_NAME);
    if (!left || !right)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_elementwise_mult(left, right);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create multiply node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_MULTIPLY_H */
