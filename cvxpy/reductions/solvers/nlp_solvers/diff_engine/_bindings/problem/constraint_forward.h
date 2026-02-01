#ifndef PROBLEM_CONSTRAINT_FORWARD_H
#define PROBLEM_CONSTRAINT_FORWARD_H

#include "common.h"

static PyObject *py_problem_constraint_forward(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *u_obj;
    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &u_obj))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    problem_constraint_forward(prob, (const double *) PyArray_DATA(u_array));
    Py_DECREF(u_array);

    PyObject *out = NULL;
    if (prob->total_constraint_size > 0)
    {
        npy_intp size = prob->total_constraint_size;
        out = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
        if (!out)
        {
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject *) out), prob->constraint_values,
               size * sizeof(double));
    }
    else
    {
        npy_intp size = 0;
        out = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    }

    return out;
}

#endif /* PROBLEM_CONSTRAINT_FORWARD_H */
