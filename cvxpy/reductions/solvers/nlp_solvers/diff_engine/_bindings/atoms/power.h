#ifndef ATOM_POWER_H
#define ATOM_POWER_H

#include "common.h"
#include "elementwise_univariate.h"

static PyObject *py_make_power(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    double p;
    if (!PyArg_ParseTuple(args, "Od", &child_capsule, &p))
    {
        return NULL;
    }
    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_power(child, p);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create power node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_POWER_H */
