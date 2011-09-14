# intended to implement a power-law fitting routine as specified in.....
# http://www.santafe.edu/~aaronc/powerlaws/
#
# The MLE for the power-law alpha is very easy to derive given knowledge
# of the lowest value at which a power law holds, but that point is 
# difficult to derive and must be acquired iteratively.

"""
plfit.py - a python power-law fitter based on code by Aaron Clauset
http://www.santafe.edu/~aaronc/powerlaws/
http://arxiv.org/abs/0706.1062 "Power-law distributions in empirical ksvalsa" 
Requires pylab (matplotlib), which requires numpy

example use:
from plfit import plfit

MyPL = plfit(myksvalsa)
MyPL.plotpdf(log=True)


"""

#from pylab import *
cimport c_numpy as cnp
cnp.import_array()
import numpy
cimport numpy
import cmath
import cython
import time
DTYPE=numpy.float
ctypedef numpy.float_t DTYPE_t

cdef extern from "math.h":
    float log(float theta)
    float sqrt(float theta)
    float pow(float x,float y)

@cython.boundscheck(False)
def plfit_loop(z,zunique=None,argunique=None,nosmall=True):
    """
    The internal loop of plfit.  Useful because it can
    be wrapped into plfit.py for direct comparison
    with the fortran version
    """
    if zunique is None or argunique is None:
        zunique,argunique = numpy.unique1d(z,return_index=True)
    cdef int lxm = len(z)
    cdef int nu = len(zunique)
    cdef cnp.ndarray[DTYPE_t,ndim=1,mode='c'] ksvals = numpy.zeros(nu) 
    cdef cnp.ndarray[DTYPE_t,ndim=1,mode='c'] av     = numpy.zeros(nu)
    cdef cnp.ndarray[DTYPE_t,ndim=1,mode='c'] data   = numpy.sort(z)
    cdef cnp.ndarray[DTYPE_t,ndim=1,mode='c'] czunique = zunique
    cdef cnp.ndarray[DTYPE_t,ndim=1,mode='c'] cargunique = argunique.astype('float')
    cdef float cx,cf,val,a,xmin,denom,nj,nk
    cdef int xm,i,iu
    for iu from 0 <= iu < nu:
        xm = <int>(argunique[iu])
        xmin = czunique[iu]
        # estimate alpha using direct MLE
        denom = 0
        for i from xm <= i < lxm:
            denom += log(data[i]/xmin)
        nj=<float>(lxm-xm)
        a =  nj / denom 
        if nosmall and ( ((a-1)/sqrt(nj+1)) >= 0.1 ):
            # 4. For continuous data, PLFIT can return erroneously large estimates of 
            #    alpha when xmin is so large that the number of obs x >= xmin is very 
            #    small. To prevent this, we can truncate the search over xmin values 
            #    before the finite-size bias becomes significant by calling PLFIT as
            #av = av[:xm] # this should only be called once
            #ksvals = ksvals[:xm] # this should only be called once
            break
        av[iu] = a
        for i from xm <= i < lxm:
            # compute KS statistic
            cx   = <float>(i-xm)/(nj)  #data; cheap float conversion?
            cf   = 1.0-pow((xmin/data[i]),a)  # fitted
            val = (cf-cx)
            if val < 0:  # math.h's abs() does not work
                val *= -1.0 
            if ksvals[iu] < val:
                ksvals[iu] = val
#        print a,nj,denom,iu,xm,(a-1)/sqrt(nj),ksvals[xm],val
    
    return ksvals,av
