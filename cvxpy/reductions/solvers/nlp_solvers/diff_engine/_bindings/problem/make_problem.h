#ifndef PROBLEM_MAKE_H
#define PROBLEM_MAKE_H

#include "common.h"

static PyObject *py_make_problem(PyObject *self, PyObject *args)
{
    PyObject *obj_capsule;
    PyObject *constraints_list;
    int verbose = 1;
    if (!PyArg_ParseTuple(args, "OO|p", &obj_capsule, &constraints_list, &verbose))
    {
        return NULL;
    }

    expr *objective = (expr *) PyCapsule_GetPointer(obj_capsule, EXPR_CAPSULE_NAME);
    if (!objective)
    {
        PyErr_SetString(PyExc_ValueError, "invalid objective capsule");
        return NULL;
    }

    if (!PyList_Check(constraints_list))
    {
        PyErr_SetString(PyExc_TypeError, "constraints must be a list");
        return NULL;
    }

    Py_ssize_t n_constraints = PyList_Size(constraints_list);
    expr **constraints = NULL;
    if (n_constraints > 0)
    {
        constraints = (expr **) malloc(n_constraints * sizeof(expr *));
        if (!constraints)
        {
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < n_constraints; i++)
        {
            PyObject *c_capsule = PyList_GetItem(constraints_list, i);
            constraints[i] =
                (expr *) PyCapsule_GetPointer(c_capsule, EXPR_CAPSULE_NAME);
            if (!constraints[i])
            {
                free(constraints);
                PyErr_SetString(PyExc_ValueError, "invalid constraint capsule");
                return NULL;
            }
        }
    }

    problem *prob =
        new_problem(objective, constraints, (int) n_constraints, verbose);
    free(constraints);

    if (!prob)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create problem");
        return NULL;
    }

    return PyCapsule_New(prob, PROBLEM_CAPSULE_NAME, problem_capsule_destructor);
}

#endif /* PROBLEM_MAKE_H */
