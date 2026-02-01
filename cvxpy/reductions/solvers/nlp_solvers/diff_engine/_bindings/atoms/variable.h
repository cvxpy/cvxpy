#ifndef ATOM_VARIABLE_H
#define ATOM_VARIABLE_H

#include "common.h"

static PyObject *py_make_variable(PyObject *self, PyObject *args)
{
    int d1, d2, var_id, n_vars;
    if (!PyArg_ParseTuple(args, "iiii", &d1, &d2, &var_id, &n_vars))
    {
        return NULL;
    }

    expr *node = new_variable(d1, d2, var_id, n_vars);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create variable node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_VARIABLE_H */
