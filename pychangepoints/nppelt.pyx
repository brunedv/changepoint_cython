import numpy as np

cimport numpy as np

import math

cimport cython
from libc.math cimport M_PI, fmax, isnan, log, sqrt

DTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

@cython.wraparound(False)
@cython.boundscheck(False)    
def cnp_pelt(np.ndarray[DTYPE_t, ndim=2] sumstat, double pen, int minseglen, int n, int nquantiles, str method):
    #cdef int n = sumstat.shape[0] - 1
    cdef ITYPE_t error = 0
    cdef ITYPE_t nchecklist

    cdef np.ndarray[ITYPE_t, ndim=1] cptsout, lastchangecpts, checklist, tmpt, numchangecpts
    a = np.zeros(n+1, dtype=np.int64)
    cptsout = a
    b = np.zeros(n+1, dtype=np.int64)

    lastchangecpts = b
    c = np.zeros(n+1 ,dtype=np.int64)
    checklist = c
    d = np.zeros(n+1, dtype=np.int64)
    tmpt = d
    e = np.zeros(n+1, dtype=np.int64)

    numchangecpts = e
    cdef double [:] lastchangelike, tmplike
    zeros_array_double = np.zeros(n+1)
    lastchangelike = zeros_array_double
    zeros_array_double_2 = np.zeros(n+1)

    tmplike = zeros_array_double_2
    cdef ITYPE_t tstar, i, whichout, nchecktmp, j, isum
    cdef double minout
    lastchangelike[0] = -pen
    lastchangecpts[0] = 0
    numchangecpts[0] = 0
    cdef np.ndarray[DTYPE_t, ndim=1] sumstatout = np.zeros(nquantiles)
    
    cdef DTYPE_t Fkl
    cdef DTYPE_t temp_cost
    cdef DTYPE_t cost
    cdef ITYPE_t nseg, isum_d

    for j in range(minseglen, 2*minseglen):
        for isum in range(0, nquantiles):
            sumstatout[isum] = sumstat[isum, j]-sumstat[isum, 0]
        cost = 0
        temp_cost = 0
        nseg = j - 0

        for isum_d in range(0, nquantiles):
            Fkl = (sumstatout[isum_d])/(nseg)
            temp_cost = (j-0)*(Fkl*log(Fkl)+(1-Fkl)*log(1-Fkl))
            if(isnan(temp_cost)):
                cost = cost
            else:
                cost = cost + temp_cost
        cost = -2*(log(2*n-1))*cost/(nquantiles)

        lastchangelike[j] = cost
    for j in range(minseglen, 2*minseglen):
        numchangecpts[j] = 1

    nchecklist = 2
    checklist[0] = 0
    checklist[1] = minseglen

    for tstar in range(2*minseglen, n+1):
        for i in range(0, nchecklist):
            for isum in range(0, nquantiles):
                sumstatout[isum] = sumstat[isum, tstar] - sumstat[isum, checklist[i]]
            tmplike[i] = lastchangelike[checklist[i]] +pen
            cost = 0
            temp_cost = 0
            nseg = tstar -  checklist[i]

            for isum_d in range(0, nquantiles):
                Fkl = (sumstatout[isum_d])/(nseg)
                temp_cost = (tstar- checklist[i])*(Fkl*log(Fkl)+(1-Fkl)*log(1-Fkl))
                if(isnan(temp_cost)):
                    cost = cost
                else:
                    cost = cost + temp_cost

            cost = -2*(log(2*n-1))*cost/(nquantiles)
            #current_cost(sumstatout, tstar, checklist[i], nquantiles, n)
            tmplike[i] += cost
        minout = tmplike[0]
        whichout = 0
        for i in range(1, nchecklist):
            if tmplike[i] <= minout:
                minout = tmplike[i]
                whichout = i
        lastchangelike[tstar] = minout
        lastchangecpts[tstar] = checklist[whichout]
        numchangecpts[tstar] = numchangecpts[lastchangecpts[tstar]]+1
        nchecktmp = 0
        for i in range(0, nchecklist):
            if(tmplike[i] <= (lastchangelike[tstar]+pen)):
                checklist[nchecktmp] = checklist[i]
                nchecktmp = nchecktmp+1
        nchecklist = nchecktmp
        checklist[nchecklist] = tstar-(minseglen-1)
        nchecklist+=1

    cdef int ncpts = 0
    cdef int last = n
    while (last != 0):
        cptsout[ncpts] = last
        last = lastchangecpts[last]
        ncpts += 1
    return np.array(cptsout)[0:ncpts], ncpts


                                  
