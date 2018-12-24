import numpy as np 
cimport numpy as np 
import math 
cimport cython

DTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

from libc.math cimport sqrt
cimport cost_function
from cost_function cimport mll_nonparametric_ed, mll_nonparametric_ed_mbic
cimport utils
from utils cimport order_vec
@cython.wraparound(False)
@cython.boundscheck(False)    
def cnp_pelt( np.ndarray[DTYPE_t, ndim=2] sumstat, double pen, int minseglen, int n, int nquantiles, str method):
    #cdef int n = sumstat.shape[0] - 1
    cdef int error = 0
    cdef int nchecklist

    cdef int [:] cptsout, lastchangecpts, checklist, tmpt, numchangecpts
    a = np.zeros(n+1,dtype=np.intc)
    cptsout = a
    b = np.zeros(n+1,dtype=np.intc)

    lastchangecpts = b
    c= np.zeros(n+1,dtype=np.intc)
    checklist = c
    d = np.zeros(n+1,dtype=np.intc)
    tmpt = d
    e = np.zeros(n+1,dtype=np.intc)

    numchangecpts = e
    cdef double [:] lastchangelike, tmplike
    zeros_array_double = np.zeros(n+1)
    lastchangelike = zeros_array_double
    zeros_array_double_2 = np.zeros(n+1)

    tmplike = zeros_array_double_2
    cdef int tstar, i, whichout, nchecktmp, j, isum
    cdef double minout
    lastchangelike[0] = -pen
    lastchangecpts[0] = 0
    numchangecpts[0] = 0
    cdef np.ndarray[DTYPE_t, ndim=1] sumstatout = np.zeros(nquantiles)
    if method =="nonparametric_ed":
        current_cost = mll_nonparametric_ed
    elif method == "mbic_nonparametric_ed":
        current_cost = mll_nonparametric_ed_mbic
    else:
        current_cost = mll_nonparametric_ed_mbic
    
    for j in range(minseglen,2*minseglen):
        for isum in range(0,nquantiles):
            sumstatout[isum]=sumstat[isum,j]-sumstat[isum,0]
        lastchangelike[j] = current_cost(sumstatout,j,0,nquantiles,n)
    for j in range(minseglen,2*minseglen):
        numchangecpts[j] = 1
    cdef DTYPE_t a1 , a2, a3

    nchecklist = 2
    checklist[0] = 0
    checklist[1] = minseglen
    for tstar in range(2*minseglen, n+1):
        for i in range(0,nchecklist):
            for isum in range(0,nquantiles):
                sumstatout[isum]= sumstat[isum,tstar] - sumstat[isum,checklist[i]]
            tmplike[i]=lastchangelike[checklist[i]] + current_cost(sumstatout, tstar, checklist[i], nquantiles, n)+pen
        minout = tmplike[0]
        whichout = 0
        for i in range(1,nchecklist):
            if tmplike[i]<= minout:
                minout=tmplike[i]
                whichout=i
        lastchangelike[tstar]=minout
        lastchangecpts[tstar]=checklist[whichout]
        numchangecpts[tstar]=numchangecpts[lastchangecpts[tstar]]+1
        nchecktmp=0
        for i in range(0,nchecklist):
            if(tmplike[i]<= (lastchangelike[tstar]+pen)):
                checklist[nchecktmp]=checklist[i]
                nchecktmp=nchecktmp+1
        nchecklist = nchecktmp
        checklist[nchecklist]=tstar-(minseglen-1)
        nchecklist+=1


    cdef int ncpts=0
    cdef int last=n
    while (last !=0):
        cptsout[ncpts] = last
        last=lastchangecpts[last]
        ncpts+=1
    return np.array(cptsout)[0:ncpts],ncpts


                                  
