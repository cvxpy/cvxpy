#ifndef ATOM_HSTACK_H
#define ATOM_HSTACK_H

#include "common.h"

static PyObject *py_make_hstack(PyObject *self, PyObject *args)
{
    PyObject *list_obj;
    if (!PyArg_ParseTuple(args, "O", &list_obj))
    {
        return NULL;
    }
    if (!PyList_Check(list_obj))
    {
        PyErr_SetString(PyExc_TypeError,
                        "First argument must be a list of expr capsules");
        return NULL;
    }
    Py_ssize_t n_args = PyList_Size(list_obj);
    if (n_args == 0)
    {
        PyErr_SetString(PyExc_ValueError, "List of expr capsules cannot be empty");
        return NULL;
    }
    expr **expr_args = (expr **) calloc(n_args, sizeof(expr *));
    for (Py_ssize_t i = 0; i < n_args; ++i)
    {
        PyObject *item = PyList_GetItem(list_obj, i);
        expr *e = (expr *) PyCapsule_GetPointer(item, EXPR_CAPSULE_NAME);
        if (!e)
        {
            free(expr_args);
            PyErr_SetString(PyExc_ValueError, "Invalid expr capsule in list");
            return NULL;
        }
        expr_args[i] = e;
    }
    int n_vars = expr_args[0]->n_vars;
    expr *node = new_hstack(expr_args, (int) n_args, n_vars);
    free(expr_args);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create hstack node");
        return NULL;
    }
    expr_retain(node);
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif // ATOM_HSTACK_H
