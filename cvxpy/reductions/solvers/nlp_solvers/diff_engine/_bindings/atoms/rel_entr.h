// SPDX-License-Identifier: Apache-2.0

#ifndef ATOM_REL_ENTR_H
#define ATOM_REL_ENTR_H

#include "bivariate.h"
#include "common.h"

/* rel_entr: rel_entr(x, y) = x * log(x/y) elementwise */
static PyObject *py_make_rel_entr(PyObject *self, PyObject *args)
{
    (void) self;
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

    expr *node = new_rel_entr_vector_args(left, right);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create rel_entr node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_REL_ENTR_H */
