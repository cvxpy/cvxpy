#ifndef PROBLEM_GRADIENT_H
#define PROBLEM_GRADIENT_H

#include "common.h"

static PyObject *py_problem_gradient(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    if (!PyArg_ParseTuple(args, "O", &prob_capsule))
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

    problem_gradient(prob);

    npy_intp size = prob->n_vars;
    PyObject *out = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *) out), prob->gradient_values,
           size * sizeof(double));

    return out;
}

#endif /* PROBLEM_GRADIENT_H */
