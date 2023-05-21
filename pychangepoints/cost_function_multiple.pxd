import numpy as np

cimport numpy as np

import math

cimport cython

DTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

from libc.math cimport M_PI, fmax, isnan, log, sqrt


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline DTYPE_t mll_mean(DTYPE_t x, DTYPE_t x2 , DTYPE_t x3, ITYPE_t n, ITYPE_t dim) nogil:
    return (x2-(x*x)*1/(n))

@cython.wraparound(False)
@cython.boundscheck(False) 
cdef inline DTYPE_t mbic_mean(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n, ITYPE_t dim) nogil:
    return (x2-(x*x)*1/(n)+log(n))

@cython.wraparound(False)
@cython.boundscheck(False) 
cdef inline DTYPE_t mll_mean_vect(np.ndarray[DTYPE_t, ndim=1] x , np.ndarray[DTYPE_t, ndim=1] x2, np.ndarray[DTYPE_t, ndim=1] x3, ITYPE_t n, ITYPE_t dim):
    cdef DTYPE_t cost =0
    cdef ITYPE_t j
    """
    for j in range(0,dim):
        cost += x2[j]-(x[j]*x[j])*1/(n)
    """
    return x2[0]-(x[0]*x[0])*1/(n)

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