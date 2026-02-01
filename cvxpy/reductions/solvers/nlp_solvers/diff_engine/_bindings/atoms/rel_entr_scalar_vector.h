// SPDX-License-Identifier: Apache-2.0

#ifndef ATOM_REL_ENTR_SCALAR_VECTOR_H
#define ATOM_REL_ENTR_SCALAR_VECTOR_H

#include "bivariate.h"
#include "common.h"

/* rel_entr_scalar_vector: rel_entr(x, y) where x is scalar, y is vector */
static PyObject *py_make_rel_entr_scalar_vector(PyObject *self, PyObject *args)
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

    expr *node = new_rel_entr_first_arg_scalar(left, right);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create rel_entr_scalar_vector node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_REL_ENTR_SCALAR_VECTOR_H */
