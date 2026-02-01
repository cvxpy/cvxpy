#ifndef ATOM_CONST_VECTOR_MULT_H
#define ATOM_CONST_VECTOR_MULT_H

#include "bivariate.h"
#include "common.h"

/* Constant vector elementwise multiplication: a ∘ f(x) where a is a constant vector
 */
static PyObject *py_make_const_vector_mult(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "OO", &child_capsule, &a_obj))
    {
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    PyArrayObject *a_array =
        (PyArrayObject *) PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!a_array)
    {
        return NULL;
    }

    /* Verify size matches child size */
    int a_size = (int) PyArray_SIZE(a_array);
    if (a_size != child->size)
    {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "vector 'a' size must match child size");
        return NULL;
    }

    double *a_data = (double *) PyArray_DATA(a_array);

    expr *node = new_const_vector_mult(a_data, child);

    Py_DECREF(a_array);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create const_vector_mult node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_CONST_VECTOR_MULT_H */
