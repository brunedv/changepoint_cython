import numpy as np 
cimport numpy as np 
import math 
cimport cython

DTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline order_vec(np.ndarray[ITYPE_t, ndim=1] a, ITYPE_t n):   
    cdef ITYPE_t i, j
    cdef ITYPE_t  t
    for i in range(0, n):
        for j in range(1, n-i):
            if(a[j-1] > a[j]):
                t = a[j-1]
                a[j-1] = a[j]
                a[j] = t
