// SPDX-License-Identifier: Apache-2.0
#ifndef ATOM_PROD_AXIS_ZERO_H
#define ATOM_PROD_AXIS_ZERO_H

#include "common.h"
#include "other.h"

static PyObject *py_make_prod_axis_zero(PyObject *self, PyObject *args)
{
    (void) self;
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

    expr *node = new_prod_axis_zero(child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create prod_axis_zero node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_PROD_AXIS_ZERO_H */
