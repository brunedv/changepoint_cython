import numpy as np

cimport numpy as np

import math

cimport cython
from .cost_function cimport (
    mbic_mean,
    mbic_meanvar,
    mbic_meanvar_exp,
    mbic_meanvar_poisson,
    mbic_var,
    mll_mean,
    mll_meanvar,
    mll_meanvar_exp,
    mll_meanvar_poisson,
    mll_nonparametric_ed,
    mll_nonparametric_ed_mbic,
    mll_var,
)
from libc.math cimport M_PI, fmax, isnan, log, sqrt
from .utils cimport order_vec

from cython.parallel import prange

DTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t


"""
def cpelt2 (np.ndarray[DTYPE_t, ndim=3] sumstat, double pen, int minseglen):
    cdef int n = sumstat.shape[0] 
    cdef int error = 0
    cdef int nchecklist

    cdef int [:] cptsout = np.zeros(n+1).astype(int)
    cdef int [:] lastchangecpts = np.zeros(2*(n + 1)).astype(int)
    cdef double [:] lastchangelike = np.zeros(n+1)
    cdef int [:] checklist = np.zeros(n+1).astype(int)
    cdef double [:] tmplike = np.zeros(n+1)
    cdef int [:] tmpt = np.zeros(n+1).astype(int)
    cdef int [:] numchangecpts = np.zeros(n+1).astype(int)
    cdef int tstar,i,j,k,whichout,nchecktmp
    cdef double minout

    lastchangelike[0] = -pen
    lastchangecpts[0] = 0
    lastchangecpts[n + 0] = 0
    lastchangecpts[n + 1] = 1
    lastchangecpts[n + 2] = 2
    lastchangecpts[n + 3] = 3
    for j in range(minseglen, 2 * minseglen):
        lastchangelike[j] = mll_mean(sumstat[j, 0], sumstat[j, 1], j)
        lastchangecpts[j] = 0
    nchecklist = 2
    checklist[0] = 0
    checklist[1] = minseglen
    for tstar in range(2 * minseglen, n + 1):
        for i in range(0,nchecklist):
            print(tstar-checklist[i])
            tmplike[i]=lastchangelike[checklist[i]] + mll_mean(sumstat[tstar,0]-sumstat[checklist[i],0],sumstat[tstar,1]-sumstat[checklist[i],1], tstar-checklist[i])+pen
        whichout = np.argmin(tmplike[0:nchecklist])
        minout = tmplike[whichout]
        lastchangelike[tstar] = minout
        lastchangecpts[tstar] = checklist[whichout]
        lastchangecpts[n + tstar] = tstar
        nchecktmp = 0
        for k in range(0,nchecklist):
            if (tmplike[k]<=lastchangelike[tstar] + pen):
                checklist[nchecktmp]=checklist[k]
                nchecktmp+=1
        checklist[nchecktmp] = (tstar - (minseglen - 1))
        nchecktmp += 1
        nchecklist = nchecktmp

    cdef int ncpts = 0
    cdef int last = n
    changepoints = np.array(0)
    while (last != 0):
        if (last != n):
            changepoints = np.append(changepoints, last)
        last = int(lastchangecpts[last])
        ncpts += 1
    changepoints = np.sort(changepoints)
    return changepoints
"""
@cython.wraparound(False)
@cython.boundscheck(False)    
def cpelt(np.ndarray[DTYPE_t, ndim=1] sumstat, double pen, int minseglen, int n, str method):
    #cdef int n = sumstat.shape[0] - 1
    cdef int error = 0
    cdef int nchecklist

    cdef int [:] cptsout, lastchangecpts, checklist, tmpt, numchangecpts
    a = np.zeros(n+1, dtype=np.intc)
    cptsout = a
    b = np.zeros(n+1, dtype=np.intc)

    lastchangecpts = b
    c= np.zeros(n+1, dtype=np.intc)
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
    elif method == "mll_var":
        current_cost = mll_var
    elif method == "mll_meanvar":
        current_cost = mll_meanvar
    elif method == "mll_meanvar_exp":
        current_cost = mll_meanvar_exp
    elif method == "mll_meanvar_poisson":
        current_cost = mll_meanvar_poisson
    elif method == "mbic_var":
        current_cost = mbic_var
    elif method == "mbic_meanvar":
        current_cost = mbic_meanvar
    elif method == "mbic_mean":
        current_cost = mbic_mean
    elif method == "mbic_meanvar_exp":
        current_cost = mbic_meanvar_exp
    elif method == "mbic_meanvar_poisson":
        current_cost = mbic_meanvar_poisson
    else:
        current_cost = mbic_meanvar
    
    cdef int j
    for j in range(minseglen, 2*minseglen):
        lastchangelike[j] = current_cost(sumstat[j], sumstat[j+n+1], sumstat[j+2*(n+1)], j)
    for j in range(minseglen, 2*minseglen):
        numchangecpts[j] = 1
    for j in range(minseglen, 2*minseglen):
        lastchangecpts[j] = 1
    cdef DTYPE_t a1 , a2, a3

    nchecklist = 2
    checklist[0] = 0
    checklist[1] = minseglen
    for tstar in range(2*minseglen, n+1):
        if lastchangelike[tstar] == 0:
            for i in range(0, nchecklist+1):
                a1 = sumstat[tstar]-sumstat[checklist[i]]
                a2 = sumstat[tstar+n+1]-sumstat[checklist[i]+n+1]
                a3 = sumstat[tstar+2*(n+1)]-sumstat[checklist[i]+2*(n+1)]
                tmplike[i] = lastchangelike[checklist[i]] + current_cost(a1, a2, a3, tstar-checklist[i])+pen
            minout = tmplike[0]
            whichout = 0
            for i in range(0, nchecklist):
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
        nchecklist += 1

    cdef int ncpts = 0
    cdef int last = n
    while (last != 0):
        cptsout[ncpts] = last
        last = lastchangecpts[last]
        ncpts += 1
    return np.array(cptsout)[0:(ncpts)], ncpts


@cython.wraparound(False)
@cython.boundscheck(False)                                   
def cbin_seg(np.ndarray[DTYPE_t, ndim=2] sumstat, ITYPE_t Q, ITYPE_t minseglen, str method):
    cdef ITYPE_t n = sumstat.shape[0] - 1
    cdef np.ndarray[ITYPE_t, ndim=1] cptsout = np.zeros(Q, dtype=np.int64)
    cdef double [:] likeout = np.zeros(Q)
    cdef ITYPE_t op_cps
    cdef double [:] lambda_ = np.zeros(n)
    cdef np.ndarray[ITYPE_t, ndim=1] tau_ = np.zeros(Q+2).astype(np.int64)
    tau_[0] = 0
    tau_[1] = n
    cdef double null
    cdef double oldmax = np.inf 
    cdef ITYPE_t q, p, i, j, st, end

    if method == "mll_mean":
        current_cost = mll_mean
    elif method == "mll_var":
        current_cost = mll_var
    elif method == "mll_meanvar":
        current_cost = mll_meanvar
    elif method == "mll_meanvar_exp":
        current_cost = mll_meanvar_exp
    elif method == "mll_meanvar_poisson":
        current_cost = mll_meanvar_poisson
    elif method == "mbic_var":
        current_cost = mbic_var
    elif method == "mbic_meanvar":
        current_cost = mbic_meanvar
    elif method == "mbic_mean":
        current_cost = mbic_mean
    elif method == "mbic_meanvar_exp":
        current_cost = mbic_meanvar_exp
    elif method == "mbic_meanvar_poisson":
        current_cost = mbic_meanvar_poisson
    else:
        current_cost = mll_meanvar

    for q in range(Q):
        lambda_= np.zeros(n)
        i = 1
        st = tau_[0]+1
        end = tau_[1]
        null = (-0.5)*current_cost(sumstat[end, 0]-sumstat[st-1, 0], sumstat[end, 1]-sumstat[st-1, 1], sumstat[end, 2]-sumstat[st-1, 2], end - st + 1)
        for j in range(2, n-2):
            if (j == end):
                st = end+1
                i = i+1
                end = tau_[i]
                null = (-0.5)*current_cost(sumstat[end, 0]-sumstat[st-1, 0], sumstat[end, 1]-sumstat[st-1, 1], sumstat[end, 2]-sumstat[st-1, 2], end - st + 1)
            else:
                if (j-st >= minseglen) & (end-j >= minseglen):
                    lambda_[j] = ((-0.5)*current_cost(sumstat[j, 0]-sumstat[st-1, 0], sumstat[j, 1]-sumstat[st-1, 1], sumstat[j, 2]-sumstat[st-1, 2], j - st + 1)) + ((-0.5)*current_cost(sumstat[end, 0]-sumstat[j, 0], sumstat[end, 1]-sumstat[j, 1], sumstat[end, 2]-sumstat[j, 2], end - j)) - null
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
def cseg_neigh(np.ndarray[DTYPE_t, ndim=2] sumstat, ITYPE_t Q, str method):
    cdef ITYPE_t n = sumstat.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] all_seg = np.zeros((n, n))
    cdef int i, j, q
    cdef DTYPE_t s

    if method == "mll_mean":
        current_cost = mll_mean
    elif method == "mll_var":
        current_cost = mll_var
    elif method == "mll_meanvar":
        current_cost = mll_meanvar
    elif method == "mll_meanvar_exp":
        current_cost = mll_meanvar_exp
    elif method == "mll_meanvar_poisson":
        current_cost = mll_meanvar_poisson
    elif method == "mbic_var":
        current_cost = mbic_var
    elif method == "mbic_meanvar":
        current_cost = mbic_meanvar
    elif method == "mbic_mean":
        current_cost = mbic_mean
    elif method == "mbic_meanvar_exp":
        current_cost = mbic_meanvar_exp
    elif method == "mbic_meanvar_poisson":
        current_cost = mbic_meanvar_poisson
    else:
        current_cost = mll_meanvar
    for i in range(0, n) :
        for j in prange(i, n, nogil=True):
            all_seg[i, j] = -0.5*current_cost(sumstat[j, 0]-sumstat[i, 0], sumstat[j, 1]-sumstat[i, 1], sumstat[j, 2]-sumstat[i, 2], j-i+1)
    cdef np.ndarray[DTYPE_t, ndim=2] like_q = np.zeros((Q, n))
    like_q[0, :] = all_seg[0, :]
    cdef  np.ndarray[ITYPE_t, ndim=2] cp = np.zeros((Q, n), dtype=np.int64)
    cdef DTYPE_t max_out = -np.inf
    cdef ITYPE_t max_which
    cdef np.ndarray[DTYPE_t, ndim=1] like

    for q in range(1, Q):
        for j in range(q, n):
            like = like_q[q-1, (q-1):(j-1)]+all_seg[q:j, j]
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
        cps_Q[q, 0] = cp[q, n-1]
        for i in range(0, q):
            cps_Q[q, i+1] = cp[q-i-1, cps_Q[q, i]]
  
    criterium = -2 * like_q[:, n-2]
    op_cps = np.argmin(criterium)
    cpts = np.sort(cps_Q[op_cps, :][cps_Q[op_cps, :] > 0])
  
    return cpts

