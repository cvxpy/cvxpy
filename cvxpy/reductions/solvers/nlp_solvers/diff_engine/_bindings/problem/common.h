#ifndef PROBLEM_COMMON_H
#define PROBLEM_COMMON_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include "problem.h"

/* Also need expr types for capsule handling */
#include "../atoms/common.h"

#define PROBLEM_CAPSULE_NAME "DNLP_PROBLEM"

static void problem_capsule_destructor(PyObject *capsule)
{
    problem *prob = (problem *) PyCapsule_GetPointer(capsule, PROBLEM_CAPSULE_NAME);
    if (prob)
    {
        free_problem(prob);
    }
}

#endif /* PROBLEM_COMMON_H */
