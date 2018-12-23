import numpy as np 
cimport numpy as np 
import math 
cimport cython

DTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

from libc.math cimport sqrt,log,M_PI, fmax


@cython.wraparound(False)
@cython.boundscheck(False)
#@cython.cdivision(False) 
cdef inline DTYPE_t mll_mean( DTYPE_t x , DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
    return (x2-(x*x)*1/(n))

cdef inline DTYPE_t mll_var(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
  return(n*((log(2*M_PI)+log(fmax(x3, 0.000000001)/n)+1)))

cdef inline DTYPE_t mll_meanvar(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
  return(n*(log(2*M_PI)+log(fmax((x2-((x*x)/n))/n,0.00000000001))+1))

cdef inline DTYPE_t mll_meanvar_exp(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
  return 2*n*(log(x)-log(n))

cdef inline DTYPE_t mll_meanvar_poisson(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
  cdef DTYPE_t resultat 
  if x==0:
    resultat=0
  else:
    resultat=2*x*(log(n)-log(x))
  return(resultat)

cdef inline DTYPE_t mbic_var(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
  return(n*(log(2*M_PI)+log(fmax(x3,0.00000000001)/n)+1)+log(n))

cdef inline DTYPE_t mbic_meanvar(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
  return(n*(log(2*M_PI)+log(fmax((x2-((x*x)/n))/n,0.00000000001))+1)+log(n))

cdef inline DTYPE_t mbic_mean(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
  return(x2-(x*x)/n+log(n))

cdef inline DTYPE_t mbic_meanvar_exp(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
  return(2*n*(log(x)-log(n))+log(n))

cdef inline DTYPE_t mbic_meanvar_poisson(DTYPE_t x, DTYPE_t x2, DTYPE_t x3, ITYPE_t n):
  cdef DTYPE_t resultat 
  if x==0:
    resultat=0
  else:
    resultat=2*x*(log(n)-log(x))+log(n)
  return(resultat)
