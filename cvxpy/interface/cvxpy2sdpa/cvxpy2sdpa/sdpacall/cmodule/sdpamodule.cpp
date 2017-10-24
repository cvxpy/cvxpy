/* ********************************************************************************
   sdpamodule.cpp
   A extention module of sdpapy with C or C++

   February 2017, Miguel Paredes Quinones
   ******************************************************************************** */

#include <Python.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
using namespace std;
#include <sdpa_call.h>
using namespace sdpa;
using sdpa::Time;

// just for compatibility
#define mwSize Py_ssize_t
#define mwIndex Py_ssize_t

#define lengthOfString 10240
#define MX_DEBUG 0

/* ============================================================
   Message
   ============================================================ */
#define rMessage(message)                       \
    {cout << message << " :: line " << __LINE__ \
          << " in " << __FILE__ << endl; }

#define rError(message)                         \
    {cout << message << " :: line " << __LINE__ \
          << " in " << __FILE__ << endl;        \
        exit(false);}

/* ============================================================
   Allocate array
   ============================================================ */
#if 1
#define NewArray(val,type,number) \
  {val = NULL; \
    try{ val = new type[number]; } \
    catch(bad_alloc){ \
        rMessage("Memory Exhausted (bad_alloc)"); abort(); } \
    catch(...){ \
        rMessage("Fatal Error (related memory allocation"); abort(); } \
  }
#else
#define NewArray(val,type,number) \
  {rMessage("New Invoked"); \
   val = NULL; val = new type[number]; \
   if  (val==NULL) {rError("Over Memory");} \
  }
#endif

#define DeleteArray(val) \
  { if  (val!=NULL) { \
      delete[] val; \
      val = NULL; \
    } \
  }

/* ============================================================
   sdpa.sdpasolver(mDIM,nBLOCK,bLOCKsTRUCT,bLOCKsTYPE,c,F,option)
   ============================================================ */
static char doc_sdpasolver[] =
    "[objVal,x,X,Y, info] = sdpa.sdpasolver(bLOCKsTRUCT,bLOCKsTYPE,c,F,option)";

static PyObject* sdpasolver(PyObject* self, PyObject* args, PyObject* kwrds)
{
    char* kwlist[] = {(char*)"bLOCKsTRUCT", (char*)"bLOCKsTYPE", (char*)"c", (char*)"F", (char*)"option", NULL};

    PyObject* bLOCKsTRUCT_ptr = NULL;
    PyObject* bLOCKsTYPE_ptr = NULL;
    PyObject* c_ptr = NULL;
    PyObject* F_ptr = NULL;
    PyDictObject* option_ptr = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kwrds, "OOOOO", kwlist,
                                    &bLOCKsTRUCT_ptr, &bLOCKsTYPE_ptr, &c_ptr, &F_ptr, &option_ptr)){
        Py_RETURN_NONE;
    }

    time_t ltime;
    time(&ltime);
    char string_time[1024];
    strcpy(string_time, ctime(&ltime));
    string_time[strlen(string_time) - 1] = '\0';

    SDPA sdpa;

    int maxIteration = 0;
    int nSymmChk = 0;
    int nDimacs = 0;

    /* strings for phase value */
    const char* szPhase[] = {
        "noINFO", "pFEAS", "dFEAS", "pdFEAS", "pdINF",
        "pFEAS_dINF", "pINF_dFEAS", "pdOPT", "pUNBD", "dUNBD"};

    /* output file */
    char* outfile = NULL;
    FILE* fp = NULL;
    FILE* fpResult = NULL;
    int nOutfile = 0;

    mwSize mDim;
    mwSize nBlock;

    /* temporary variables */
    mwIndex k;
    int size;
    double* tmp_ptr = NULL;
    char* tmpPrint = NULL;
    PyObject* tmpObj;

    TimeStart(SDPA_START);
    TimeStart(SDPA_CONVERT_START);

#if MX_DEBUG
    rMessage("");
#endif

    /* --------------------------------------------------
       Set SDPA parameters by OPTIONS
       -------------------------------------------------- */
    /* Max Iteration */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "maxIteration");
    
    #if PY_MAJOR_VERSION >= 3
    maxIteration = (int)PyLong_AsLong(tmpObj);
    #else
    maxIteration = (int)PyInt_AsLong(tmpObj);
    #endif
    
    sdpa.setParameterMaxIteration(maxIteration);

    /* epsilonStar */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "epsilonStar");
    sdpa.setParameterEpsilonStar(PyFloat_AsDouble(tmpObj));

    /* lambdaStar */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "lambdaStar");
    sdpa.setParameterLambdaStar(PyFloat_AsDouble(tmpObj));

    /* omegaStar */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "omegaStar");
    sdpa.setParameterOmegaStar(PyFloat_AsDouble(tmpObj));

    /* lowerBound */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "lowerBound");
    sdpa.setParameterLowerBound(PyFloat_AsDouble(tmpObj));

    /* upperBound */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "upperBound");
    sdpa.setParameterUpperBound(PyFloat_AsDouble(tmpObj));

    /* betaStar */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "betaStar");
    sdpa.setParameterBetaStar(PyFloat_AsDouble(tmpObj));

    /* betaBar */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "betaBar");
    sdpa.setParameterBetaBar(PyFloat_AsDouble(tmpObj));

    /* gammaStar */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "gammaStar");
    sdpa.setParameterGammaStar(PyFloat_AsDouble(tmpObj));

    /* epsilonDash */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "epsilonDash");
    sdpa.setParameterEpsilonDash(PyFloat_AsDouble(tmpObj));

    /* isSymmetric */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "isSymmetric");
    
#if PY_MAJOR_VERSION >= 3
    nSymmChk = (int)PyLong_AsLong(tmpObj);
#else
    nSymmChk = (int)PyInt_AsLong(tmpObj);
#endif
    /* isDimacs */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "isDimacs");
#if PY_MAJOR_VERSION >= 3
    nDimacs = (int)PyLong_AsLong(tmpObj);
#else
    nDimacs = (int)PyInt_AsLong(tmpObj);
#endif
    /* yPrint */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "yPrint");
#if PY_MAJOR_VERSION >= 3
    tmpPrint = PyBytes_AsString(PyUnicode_AsUTF8String(tmpObj));
#else
    tmpPrint = PyString_AsString(tmpObj);
#endif
    sdpa.setParameterPrintXVec(tmpPrint);

    /* sPrint */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "sPrint");
#if PY_MAJOR_VERSION >= 3
    tmpPrint = PyBytes_AsString(PyUnicode_AsUTF8String(tmpObj));
#else
    tmpPrint = PyString_AsString(tmpObj);
#endif
    sdpa.setParameterPrintXMat(tmpPrint);

    /* xPrint */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "xPrint");
#if PY_MAJOR_VERSION >= 3
    tmpPrint = PyBytes_AsString(PyUnicode_AsUTF8String(tmpObj));
#else
    tmpPrint = PyString_AsString(tmpObj);
#endif
    sdpa.setParameterPrintYMat(tmpPrint);

    /* infPrint */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "infPrint");
#if PY_MAJOR_VERSION >= 3
    tmpPrint = PyBytes_AsString(PyUnicode_AsUTF8String(tmpObj));
#else
    tmpPrint = PyString_AsString(tmpObj);
#endif
    sdpa.setParameterPrintInformation(tmpPrint);

    /* print */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "print");
#if PY_MAJOR_VERSION >= 3
    outfile = PyBytes_AsString(PyUnicode_AsUTF8String(tmpObj));
#else
    outfile = PyString_AsString(tmpObj);
#endif

    /* default setting is displaying information to stdout */
    fp = fp;
    if (strlen(outfile) == 0) {
        fp = NULL;
    } else {
        if (strncmp("display", outfile, strlen(outfile)) == 0) {
            fp = stdout;
        } else if (strncmp("no", outfile, strlen(outfile)) == 0) {
            fp = NULL;
        } else {
            fp = fopen(outfile, "at");
            if (fp == NULL) {
                printf("Failed to open %s\n", outfile);
                fp = stdout;
            } else {
                nOutfile = 1;
            }
        }
    }
    sdpa.setDisplay(fp);

    /* resultFile */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "sdpaResult");
#if PY_MAJOR_VERSION >= 3
    outfile = PyBytes_AsString(PyUnicode_AsUTF8String(tmpObj));
#else
    tmpPrint = PyString_AsString(tmpObj);
#endif
    fpResult = NULL;
    if (strlen(outfile) > 0) {
        if(strncmp("no", outfile, strlen(outfile)) == 0) {
            // printf("resultFile is NULL\n");
        } else {
            fpResult = fopen(outfile, "w");
            if (fpResult == NULL) {
                printf("Failed to open %s\n", outfile);
                printf("Skip the detail file\n");
            }
        }
    }
    sdpa.setResultFile(fpResult);

    if (fp) {
        fprintf( fp,"SDPA start at [%s]\n",string_time );
    }
    if (fpResult) {
        fprintf( fpResult,"SDPA start at [%s]\n",string_time );
    }

    /* numThreads */
    tmpObj = PyDict_GetItemString((PyObject*)option_ptr, "numThreads");
#if PY_MAJOR_VERSION >= 3
    sdpa.setNumThreads((int)PyLong_AsLong(tmpObj));
#else
    sdpa.setNumThreads((int)PyInt_AsLong(tmpObj));
#endif

#if MX_DEBUG
    rMessage("");
#endif

    /* --------------------------------------------------
       initialize SDPA class members
       -------------------------------------------------- */

    tmpObj = PyObject_GetAttrString(c_ptr, "size_row");
#if PY_MAJOR_VERSION >= 3
    mDim = PyLong_AsLong(tmpObj);
#else
    mDim = PyInt_AsLong(tmpObj);
#endif
    nBlock = PyList_Size(bLOCKsTRUCT_ptr);

    PyObject* objVal_ptr = NULL;
    PyObject* x_ptr = NULL;
    PyObject* X_ptr = NULL;
    PyObject* Y_ptr = NULL;
    PyDictObject* info_ptr = NULL;

    objVal_ptr = PyList_New(2);
    x_ptr = PyList_New(mDim);
    X_ptr = PyList_New(nBlock);
    Y_ptr = PyList_New(nBlock);
    info_ptr = (PyDictObject*)PyDict_New();
    if ( objVal_ptr == NULL || x_ptr == NULL || X_ptr == NULL || Y_ptr == NULL || info_ptr == NULL) {
        cout << "Memory Over for Solution Space" << endl;
        Py_XDECREF(objVal_ptr);
        Py_XDECREF(x_ptr);
        Py_XDECREF(X_ptr);
        Py_XDECREF(Y_ptr);
        Py_XDECREF(info_ptr);
        return PyErr_NoMemory();
    }
    sdpa.inputConstraintNumber(mDim);

#if MX_DEBUG
    rMessage("");
#endif

    sdpa.inputBlockNumber(nBlock);

#if MX_DEBUG
    rMessage("");
#endif

    /* get bLOCKsTRUCT and bLOCKsTYPE */

    int* bLOCKsTRUCT = NULL;
    int* bLOCKsTYPE = NULL;

    int nCones = (int)PyList_Size(bLOCKsTYPE_ptr);

    if (nBlock != nCones){
    rError("Size of block types different than size of blocks struct");
    }

    if (nBlock > 0) {
        NewArray(bLOCKsTRUCT, int, nBlock);
        NewArray(bLOCKsTYPE, int, nBlock);

        for (int block = 0; block < nBlock; block++) {
#if PY_MAJOR_VERSION >= 3 
            bLOCKsTRUCT[block]= PyLong_AsLong(PyList_GetItem(bLOCKsTRUCT_ptr, block));
            bLOCKsTYPE[block]= PyLong_AsLong(PyList_GetItem(bLOCKsTYPE_ptr, block));
#else
	        bLOCKsTRUCT[block]= PyInt_AsLong(PyList_GetItem(bLOCKsTRUCT_ptr, block));
            bLOCKsTYPE[block]= PyInt_AsLong(PyList_GetItem(bLOCKsTYPE_ptr, block));
#endif
        }
    }

    for (int block = 0; block < nBlock; block++) {
      sdpa.inputBlockSize(block + 1, bLOCKsTRUCT[block]);

      if (bLOCKsTYPE[block] == 0){

        sdpa.inputBlockType(block + 1, SDPA::LP);

      }else if(bLOCKsTYPE[block] == 1){

        sdpa.inputBlockType(block + 1, SDPA::SDP);

      }else if(bLOCKsTYPE[block] == 2){

        sdpa.inputBlockType(block + 1, SDPA::SOCP);
      }
    }

#if MX_DEBUG
    rMessage("");
#endif

    /* --------------------------------------------------
       Execute initializeUpperTriangleSpace()
       -------------------------------------------------- */
    sdpa.initializeUpperTriangleSpace();

#if MX_DEBUG
    rMessage("");
#endif

    int num_nnz;
    double val;
    PyObject* datObj;
    PyObject* rowObj;
    PyObject* indptrObj;
    PyObject* mObj;
    PyObject* nObj;
    /* --------------------------------------------------
       Input c
       -------------------------------------------------- */
    datObj = PyObject_GetAttrString(c_ptr, "values");
    rowObj = PyObject_GetAttrString(c_ptr, "rowind");
    num_nnz = PyList_Size(datObj);


    for (int i = 0; i < num_nnz; i++) {
        val = PyFloat_AsDouble(PyList_GetItem(datObj, i));
#if PY_MAJOR_VERSION >= 3 
        int row = PyLong_AsLong(PyList_GetItem(rowObj, i));
#else
	    int row = PyInt_AsLong(PyList_GetItem(rowObj, i));
#endif
        sdpa.inputCVec(row + 1, val);
    }

#if MX_DEBUG
    rMessage("");
#endif

    /* --------------------------------------------------
       Input F
       -------------------------------------------------- */

    for (int m = 0; m <= mDim ; m++){
        mObj = PyList_GetItem(F_ptr,m);

      for (int block = 0; block < nBlock ; block++){

         nObj = PyList_GetItem(mObj,block);

        datObj = PyObject_GetAttrString(nObj, "values");
        rowObj = PyObject_GetAttrString(nObj, "rowind");
        indptrObj = PyObject_GetAttrString(nObj, "colptr");

	if ( PyList_Size(datObj) == 0 ) continue ;
        for(int j = 0 ; j < bLOCKsTRUCT[block] ; j++){

#if PY_MAJOR_VERSION >= 3 
          int iss = PyLong_AsLong(PyList_GetItem(indptrObj,j));
          int iff = PyLong_AsLong(PyList_GetItem(indptrObj,j+1));
#else
	 int iss = PyInt_AsLong(PyList_GetItem(indptrObj,j));
          int iff = PyInt_AsLong(PyList_GetItem(indptrObj,j+1));
#endif
            if ( iss == iff ) continue ;

            for(int k = iss; k < iff ; k++){
#if PY_MAJOR_VERSION >= 3 
             int i=PyLong_AsLong(PyList_GetItem(rowObj,k));
#else
             int i=PyInt_AsLong(PyList_GetItem(rowObj,k));
#endif
             if (bLOCKsTYPE[block] == 0){

                val = PyFloat_AsDouble(PyList_GetItem(datObj, k));
                sdpa.inputElement(m, block + 1, i + 1, i + 1,val);

              }else if(bLOCKsTYPE[block] == 1){

                if (i <= j) {
                  val = PyFloat_AsDouble(PyList_GetItem(datObj, k));
                  sdpa.inputElement(m, block + 1, i + 1, j + 1,val);
                }

              }else if(bLOCKsTYPE[block] == 2){

                val = PyFloat_AsDouble(PyList_GetItem(datObj, k));
                sdpa.inputElement(m, block + 1, i + 1, i + 1,val);
              }
            }
        }
      }
    }

    /* --------------------------------------------------
       Check the consistence of F, c
       -------------------------------------------------- */
    if (nSymmChk) {
        sdpa.initializeUpperTriangle(true);
    } else {
        sdpa.initializeUpperTriangle(false);
    }

#if MX_DEBUG
    rMessage("");
    
#endif
    //sdpa.writeInputSparse((char*)"b.dat-s", (char*)"%e");
    /* --------------------------------------------------
       Solve by SDPA
       -------------------------------------------------- */
#if MX_DEBUG
    rMessage("");
#endif

    sdpa.initializeSolve();

#if MX_DEBUG
    rMessage("");
#endif
    if (fp) {
        printf("Converted to SDPA internal data / ");
        printf("Starting SDPA main loop\n");
    }
    TimeEnd(SDPA_CONVERT_END);
    TimeStart(SDPA_SOLVE_START);
    sdpa.solve();
    TimeEnd(SDPA_SOLVE_END);

#if MX_DEBUG
    rMessage("");
#endif

    /* --------------------------------------------------
       Set output values to arguments
       -------------------------------------------------- */
    TimeStart(SDPA_RETRIEVE_START);
    if (fp) {
        printf("Converting optimal solution to SDPA format\n");
    }
    /* Optimal value for xVec */
    tmp_ptr = sdpa.getResultXVec();
    if (tmp_ptr != NULL) {
        for (k = 0; k < mDim; k++) {
            PyList_SetItem(x_ptr, k, PyFloat_FromDouble(tmp_ptr[k]));
        }
    }

    /* Optimal value for YMat */

    PyObject* block_vector;
    PyObject* block_matrix ;
    //PyObject* block_line ;

        for (int block = 0; block < nBlock ; block++){
            size = sdpa.getBlockSize(block+1);
            tmp_ptr = sdpa.getResultYMat(block+1);

            if ( bLOCKsTYPE[block] == 0 || bLOCKsTYPE[block] == 2 ){

                block_vector = PyList_New(size);

                for (int i = 0; i < size; ++i) {
                    PyList_SetItem(block_vector, i, PyFloat_FromDouble(tmp_ptr[i]));
                }

                PyList_SetItem(Y_ptr, block,block_vector);

            }else{

                block_matrix = PyList_New(size*(size+1)/2);
                int ind = 0 ;
                for (int i=0; i<size; ++i) {

                    for (int j=i; j<size; ++j) {

                        PyList_SetItem(block_matrix, ind, PyFloat_FromDouble(tmp_ptr[i+size*j]));
                        ind++;
                    }

                }

                PyList_SetItem(Y_ptr, block,block_matrix);

            }

        }

    /* Optimal value for XMat */
    for (int block = 0; block < nBlock ; block++){
        size = sdpa.getBlockSize(block+1);
        tmp_ptr = sdpa.getResultXMat(block+1);

        if ( bLOCKsTYPE[block] == 0 || bLOCKsTYPE[block] == 2 ){

            block_vector = PyList_New(size);

            for (int i = 0; i < size; ++i) {
                PyList_SetItem(block_vector, i, PyFloat_FromDouble(tmp_ptr[i]));

            }

            PyList_SetItem(X_ptr, block,block_vector);

        }else{

            block_matrix = PyList_New(size*(size+1)/2);

            int ind = 0 ;
            for (int i=0; i<size; ++i) {

                for (int j=i; j<size; ++j) {

                    PyList_SetItem(block_matrix, ind, PyFloat_FromDouble(tmp_ptr[i+size*j]));
                    ind++;
                }

            }

            PyList_SetItem(X_ptr, block,block_matrix);


        }

    }

    PyList_SetItem(objVal_ptr, 0, Py_BuildValue("d",sdpa.getPrimalObj()));

    PyList_SetItem(objVal_ptr, 1, Py_BuildValue("d",sdpa.getDualObj()));

    TimeEnd(SDPA_RETRIEVE_END);

    /* Dimacs Error Information */
    if (nDimacs != 0) {
        printf("Computing Dimacs Error\n");
        double dimacs_error[7];
        sdpa.getDimacsError(dimacs_error);
        PyDict_SetItemString((PyObject*)info_ptr, "dimacs",
                             Py_BuildValue("(dddddd)",
                                           dimacs_error[1],
                                           dimacs_error[2],
                                           dimacs_error[3],
                                           dimacs_error[4],
                                           dimacs_error[5],
                                           dimacs_error[6]));
    }

    /* Phase information */
    PyDict_SetItemString((PyObject*)info_ptr, "phasevalue",
                         Py_BuildValue( "s", szPhase[sdpa.getPhaseValue()]));

    /* Iteration */
    PyDict_SetItemString((PyObject*)info_ptr, "iteration",
                         Py_BuildValue("i", sdpa.getIteration()));

    /* primalObj */
    PyDict_SetItemString((PyObject*)info_ptr, "primalObj",
                         Py_BuildValue("d", sdpa.getDualObj()));

    /* dualObj */
    PyDict_SetItemString((PyObject*)info_ptr, "dualObj",
                         Py_BuildValue("d", sdpa.getPrimalObj()));

    /* primalError */
    PyDict_SetItemString((PyObject*)info_ptr, "primalError",
                         Py_BuildValue("d", sdpa.getDualError()));

    /* dualError */
    PyDict_SetItemString((PyObject*)info_ptr, "dualError",
                         Py_BuildValue("d", sdpa.getPrimalError()));

    /* digits */
    PyDict_SetItemString((PyObject*)info_ptr, "digits",
                         Py_BuildValue("d", sdpa.getDigits()));

    /* dualityGap */
    PyDict_SetItemString((PyObject*)info_ptr, "dualityGap",
                         Py_BuildValue("d", sdpa.getDualityGap()));

    /* mu */
    PyDict_SetItemString((PyObject*)info_ptr, "mu",
                         Py_BuildValue("d", sdpa.getMu()));

    /* solveTime */
    PyDict_SetItemString((PyObject*)info_ptr, "solveTime",
                         Py_BuildValue("d", TimeCal(SDPA_SOLVE_START, SDPA_SOLVE_END)));

    /* convertingTime */
    PyDict_SetItemString((PyObject*)info_ptr, "convertingTime",
                         Py_BuildValue("d", TimeCal(SDPA_CONVERT_START, SDPA_CONVERT_END)));

    /* retrivingTime */
    PyDict_SetItemString((PyObject*)info_ptr, "retrievingTime",
                         Py_BuildValue("d", TimeCal(SDPA_RETRIEVE_START, SDPA_RETRIEVE_END)));

    /* totalTime */
    TimeEnd(SDPA_END);
    PyDict_SetItemString((PyObject*)info_ptr, "sdpaTime",
                         Py_BuildValue("d", TimeCal(SDPA_START,SDPA_END)));

    time(&ltime);
    strcpy(string_time, ctime(&ltime));
    string_time[strlen(string_time) - 1] = '\0';

    if (fp) {
        fprintf(fp,"SDPA end at [%s]\n", string_time);
    }
    if (fpResult) {
        fprintf(fpResult,"SDPA end at [%s]\n", string_time);
    }

    /* close output file */
    if (nOutfile) {
        fclose(fp);
    }

#if MX_DEBUG
    rMessage("");
#endif

    /*** Free allocated memory ****/

#if MX_DEBUG
    rMessage("");
#endif

    #if MX_DEBUG
    rMessage("");
#endif

    sdpa.terminate();

#if MX_DEBUG
    rMessage("");
#endif

    return Py_BuildValue("OOOOO",objVal_ptr, x_ptr, X_ptr, Y_ptr, info_ptr);
}

/* --------------------------------------------------
   INITIALIZER of this module
   -------------------------------------------------- */
PyDoc_STRVAR(sdpa__doc__, "SDPA: SDPAP internal API.\n **** CAUTION **** \nDo NOT call them directly. Call via SDPAP module.\n");

static PyObject* sdpamodule;

static PyMethodDef sdpa_functions[] = {
    {"sdpasolver", (PyCFunction) sdpasolver, METH_VARARGS | METH_KEYWORDS, doc_sdpasolver},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef sdpadef = {
        PyModuleDef_HEAD_INIT,
        "sdpa",     /* m_name */
        sdpa__doc__,  /* m_doc */
        -1,                  /* m_size */
        sdpa_functions,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_sdpa(void)
#else

PyMODINIT_FUNC initsdpa(void)
#endif
{

#if PY_MAJOR_VERSION >= 3
    sdpamodule = PyModule_Create(&sdpadef);

#else
    sdpamodule = Py_InitModule3("sdpa", sdpa_functions, sdpa__doc__);
#endif

#if PY_MAJOR_VERSION >= 3
    return sdpamodule;
#else
    return;
#endif
};