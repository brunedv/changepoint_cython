import math

import numpy as np

cimport cython
cimport numpy as np
from .cost_function_multiple cimport mbic_mean, mll_mean, order_vec
from libc.math cimport sqrt

from cython.parallel import prange

DTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t



@cython.wraparound(False)
@cython.boundscheck(False)  
def cbin_seg_multiple(np.ndarray[DTYPE_t, ndim=3] sumstat, ITYPE_t Q, ITYPE_t minseglen, str method):
    cdef ITYPE_t n = sumstat.shape[0] - 1
    cdef ITYPE_t dim = sumstat.shape[2]
    cdef np.ndarray[ITYPE_t, ndim=1] cptsout = np.zeros(Q, dtype=np.int64)
    cdef double [:] likeout = np.zeros(Q)
    cdef ITYPE_t op_cps
    cdef double [:] lambda_ = np.zeros(n)
    cdef np.ndarray[ITYPE_t, ndim=1] tau_ = np.zeros(Q+2).astype(np.int64)
    tau_[0] = 0
    tau_[1] = n
    cdef double null, res_lamb
    cdef double oldmax = np.inf 
    cdef ITYPE_t q, p, i, j, st, end, dim_

    if method == "mll_mean":
        current_cost = mll_mean
    else:
        current_cost = mbic_mean

    for q in range(Q):
        lambda_= np.zeros(n)
        i = 1
        st = tau_[0]+1
        end = tau_[1]
        null = 0
        for dim_ in range(0, dim):
            null += (-0.5)*current_cost(sumstat[end, 0, dim_]-sumstat[st-1, 0, dim_], sumstat[end, 1, dim_]-sumstat[st-1, 1, dim_], sumstat[end, 2, dim_]-sumstat[st-1, 2, dim_], end - st + 1, dim)
        for j in range(2, n-2):
            if (j == end):
                st = end+1
                i = i+1
                end = tau_[i]
                null = 0
                for dim_ in range(0, dim):
                    null += (-0.5)*current_cost(sumstat[end, 0, dim_]-sumstat[st-1, 0, dim_], sumstat[end, 1, dim_]-sumstat[st-1, 1, dim_], sumstat[end, 2, dim_]-sumstat[st-1, 2, dim_], end - st + 1, dim)
            else:
                if (j-st >= minseglen) & (end-j >= minseglen):
                    res_lamb = 0
                    for dim_ in range(0, dim):
                        res_lamb+=((-0.5)*current_cost(sumstat[j, 0, dim_]-sumstat[st-1, 0, dim_], sumstat[j, 1, dim_]-sumstat[st-1, 1, dim_], sumstat[j, 2, dim_]-sumstat[st-1, 2, dim_], j - st + 1, dim)) + ((-0.5)*current_cost(sumstat[end, 0, dim_]-sumstat[j, 0, dim_], sumstat[end, 1, dim_]-sumstat[j, 1, dim_], sumstat[end, 2, dim_]-sumstat[j, 2, dim_], end - j, dim)) - null
                    lambda_[j] = res_lamb
        max_out = np.max(lambda_)
        whichout = np.argmax(lambda_)

        cptsout[q] = whichout
        likeout[q] =  min(oldmax, max_out)
        oldmax = likeout[q]
        tau_[q+2] = whichout
        #tau_ = order_vec(tau_, q+3)
        order_vec(tau_, q+3)

    return np.array(cptsout)

@cython.wraparound(False)
@cython.boundscheck(False) 
def cpelt_multiple(np.ndarray[DTYPE_t, ndim=3] sumstat, double pen, int minseglen, int n, str method):
    #cdef int n = sumstat.shape[0] - 1
    cdef int error = 0
    cdef int nchecklist
    cdef ITYPE_t dim = sumstat.shape[2]

    cdef int [:] cptsout, lastchangecpts, checklist, tmpt, numchangecpts
    a = np.zeros(n+1, dtype=np.intc)
    cptsout = a
    b = np.zeros(n+1, dtype=np.intc)

    lastchangecpts = b
    c = np.zeros(n+1, dtype=np.intc)
    checklist = c
    d = np.zeros(n+1, dtype=np.intc)
    tmpt = d
    e = np.zeros(n+1, dtype=np.intc)

    numchangecpts = e
    cdef double [:] lastchangelike, tmplike
    zeros_array_double = np.zeros(n+1)
    lastchangelike = zeros_array_double
    zeros_array_double_2 = np.zeros(n+1)

    tmplike = zeros_array_double_2
    cdef int tstar, i, whichout, nchecktmp
    cdef double minout
    lastchangelike[0] = -pen
    lastchangecpts[0] = 0
    numchangecpts[0] = 0
    if method == "mll_mean":
        current_cost = mll_mean
    elif method == "mbic_mean":
        current_cost = mbic_mean
    else: 
        current_cost = mbic_mean

    
    cdef int j, dim_, dim_1
    for j in range(minseglen, 2*minseglen):
        lastchangelike[j] = 0
        for dim_ in range(0, dim):
            lastchangelike[j] += current_cost(sumstat[j, 0, dim_], sumstat[j, 1, dim_], sumstat[j, 2, dim_], j, dim)
    for j in range(minseglen, 2*minseglen):
        numchangecpts[j] = 1
    cdef  DTYPE_t a1 , a2, a3

    nchecklist = 2
    checklist[0] = 0
    checklist[1] = minseglen
    for tstar in range(2*minseglen, n+1):
        if lastchangelike[tstar] == 0:
            for i in range(0, nchecklist):
                tmplike[i] = lastchangelike[checklist[i]] + pen
                for dim_1 in range(0, dim):
                    a1 = sumstat[tstar, 0, dim_1]-sumstat[checklist[i], 0, dim_1]
                    a2 = sumstat[tstar, 1, dim_1]-sumstat[checklist[i], 1, dim_1]
                    a3 = sumstat[tstar, 2, dim_1]-sumstat[checklist[i], 2, dim_1]
                    tmplike[i] += current_cost(a1, a2, a3, tstar-checklist[i], dim)
                #tmplike[i] += lastchangelike[checklist[i]] + pen
            minout = tmplike[0]
            whichout = 0
            for i in range(1, nchecklist):
                if tmplike[i] <= minout:
                    minout = tmplike[i]
                    whichout = i
            lastchangelike[tstar] = minout
            lastchangecpts[tstar] = checklist[whichout]
            numchangecpts[tstar] = numchangecpts[lastchangecpts[tstar]]+1
            nchecktmp=0
            for i in range(0, nchecklist):
                if(tmplike[i] <= (lastchangelike[tstar]+pen)):
                    checklist[nchecktmp] = checklist[i]
                    nchecktmp = nchecktmp+1
            nchecklist = nchecktmp
        checklist[nchecklist] = tstar-(minseglen-1)
        nchecklist += 1

    cdef int ncpts = 0
    cdef int last = n
    while (last != 0):
        cptsout[ncpts] = last
        last=lastchangecpts[last]
        ncpts += 1
    return np.array(cptsout)[0:ncpts], ncpts

@cython.wraparound(False)
@cython.boundscheck(False) 
def cseg_neigh_multiple(np.ndarray[DTYPE_t, ndim=3] sumstat, ITYPE_t Q, str method):
    cdef ITYPE_t n=sumstat.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] all_seg = np.zeros((n, n))
    cdef int i, j, q, dim_
    cdef DTYPE_t s
    cdef ITYPE_t dim = sumstat.shape[2]

    if method == "mll_mean":
        current_cost = mll_mean
    elif method == "mbic_mean":
        current_cost = mbic_mean
    else: 
        current_cost = mbic_mean

    for i in range(0, n) :
        for j in prange(i, n, nogil=True):
            all_seg[i,j] = 0
            for dim_ in range(0, dim):
                all_seg[i, j] += -0.5*current_cost(sumstat[j, 0, dim_]-sumstat[i, 0, dim_], sumstat[j, 1, dim_]-sumstat[i, 1, dim_], sumstat[j, 2, dim_]-sumstat[i, 2, dim_], j - i+1, dim)
    cdef np.ndarray[DTYPE_t, ndim=2] like_q = np.zeros((Q, n))
    like_q[0, :] = all_seg[0, :]
    cdef  np.ndarray[ITYPE_t, ndim=2] cp = np.zeros((Q, n), dtype=np.int64)
    cdef DTYPE_t max_out = -np.inf
    cdef ITYPE_t max_which
    cdef np.ndarray[DTYPE_t, ndim=1] like

    for q in range(1, Q):
        for j in range(q, n):
            like = like_q[q-1, (q-1):(j-1)]+all_seg[q:j,j]
            max_which = 0
            max_out = -np.inf
            for i in range(0, j-q):
                if  max_out <= like[i]:
                    max_out = like[i]
                    max_which = i
            #print(max_out, max_which)

            like_q[q, j] = max_out
            cp[q, j] = max_which + q
    cdef  np.ndarray[ITYPE_t, ndim=2] cps_Q = np.zeros((Q, Q), dtype=np.int64)
    for q in range(1, Q):
        cps_Q[q,0] = cp[q, n-1]
        for i in range(0, q):
            cps_Q[q, i+1] = cp[q-i-1, cps_Q[q, i]]
  
    criterium = -2 * like_q[:, n-2]
    op_cps = np.argmin(criterium)
    cpts = np.sort(cps_Q[op_cps, :][cps_Q[op_cps, :] > 0])
  
    return cpts
