#ifndef ATOM_CONST_SCALAR_MULT_H
#define ATOM_CONST_SCALAR_MULT_H

#include "bivariate.h"
#include "common.h"

/* Constant scalar multiplication: a * f(x) where a is a constant double */
static PyObject *py_make_const_scalar_mult(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    double a;

    if (!PyArg_ParseTuple(args, "Od", &child_capsule, &a))
    {
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_const_scalar_mult(a, child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create const_scalar_mult node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_CONST_SCALAR_MULT_H */
