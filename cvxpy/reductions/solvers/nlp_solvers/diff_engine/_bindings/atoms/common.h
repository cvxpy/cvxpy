#ifndef ATOMS_COMMON_H
#define ATOMS_COMMON_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include "affine.h"
#include "elementwise_univariate.h"
#include "expr.h"

#define EXPR_CAPSULE_NAME "DNLP_EXPR"

static void expr_capsule_destructor(PyObject *capsule)
{
    expr *node = (expr *) PyCapsule_GetPointer(capsule, EXPR_CAPSULE_NAME);
    if (node)
    {
        free_expr(node);
    }
}

#endif /* ATOMS_COMMON_H */
