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
@cython.cdivision(True)
#@cython.cdivision(False) 
cdef inline DTYPE_t mll_mean(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
    return (x2-(x*x)*1/(n))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t mll_var(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
  return(n*((log(2*M_PI)+log(fmax(x3, 0.000000001)/n)+1)))

@cython.wraparound(False)
@cython.boundscheck(False) 
@cython.cdivision(True)
cdef inline DTYPE_t mll_meanvar(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
  return(n*(log(2*M_PI)+log(fmax((x2-((x*x)/n))/n,0.00000000001))+1))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t mll_meanvar_exp(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
  return 2*n*(log(x)-log(n))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t mll_meanvar_poisson(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
  cdef DTYPE_t resultat 
  if x==0:
    resultat=0
  else:
    resultat=2*x*(log(n)-log(x))
  return(resultat)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t mbic_var(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
  return(n*(log(2*M_PI)+log(fmax(x3,0.00000000001)/n)+1)+log(n))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t mbic_meanvar(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
  return(n*(log(2*M_PI)+log(fmax((x2-((x*x)/n))/n,0.00000000001))+1)+log(n))

@cython.wraparound(False)
@cython.boundscheck(False) 
@cython.cdivision(True)
cdef inline DTYPE_t mbic_mean(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
  return(x2-(x*x)/n+log(n))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t mbic_meanvar_exp(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
  return(2*n*(log(x)-log(n))+log(n))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t mbic_meanvar_poisson(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n) nogil:
  cdef DTYPE_t resultat 
  if x == 0:
    resultat = 0
  else:
    resultat = 2*x*(log(n)-log(x))+log(n)
  return(resultat)

cdef inline  DTYPE_t mll_nonparametric_ed(np.ndarray[DTYPE_t, ndim=1] sumstatout, ITYPE_t tstar, ITYPE_t checklist, ITYPE_t nquantiles, ITYPE_t n):
  cdef DTYPE_t Fkl
  cdef DTYPE_t temp_cost
  cdef DTYPE_t cost
  cdef ITYPE_t nseg, isum

  cost = 0
  temp_cost = 0
  nseg = tstar - checklist

  for isum in range(0, nquantiles):
    Fkl = (sumstatout[isum])/(nseg)
    temp_cost = (tstar-checklist)*(Fkl*log(Fkl)+(1-Fkl)*log(1-Fkl))
    if(isnan(temp_cost)):
      cost = cost
    else:
      cost = cost + temp_cost


  cost = -2*(log(2*n-1))*cost/(nquantiles)
  return(cost)


cdef inline  DTYPE_t  mll_nonparametric_ed_mbic(np.ndarray[DTYPE_t, ndim=1] sumstatout, ITYPE_t tstar, ITYPE_t checklist, ITYPE_t nquantiles, ITYPE_t n):
  cdef DTYPE_t Fkl
  cdef DTYPE_t temp_cost
  cdef DTYPE_t cost
  cdef ITYPE_t nseg, isum

  cost = 0
  temp_cost = 0
  nseg = tstar - checklist

  for isum in range(0, nquantiles):
    Fkl = (sumstatout[isum])/(nseg)
    temp_cost = (tstar-checklist)*(Fkl*log(Fkl)+(1-Fkl)*log(1-Fkl))
    if(isnan(temp_cost)):
      cost = cost
    else:
      cost = cost + temp_cost
  cost = -2*(log(2*n-1))*cost/(nquantiles)
  return(cost)

