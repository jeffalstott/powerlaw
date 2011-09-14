import cplfit
import plfit
import time
from numpy.random import rand,seed
from numpy import unique,sort,array,asarray,log,sum,min,max,argmin,argmax,arange
import sys
ne = int(sys.argv[1])
seed(1)
X=plfit.plexp_inv(rand(ne),1,2.5)
X[:100] = X[100:200]
t1=time.time(); p3=plfit.plfit(X,usefortran=False,usecy=True); print time.time()-t1
t1=time.time(); p1=plfit.plfit(X); print time.time()-t1
t1=time.time(); p3=plfit.plfit(X,usefortran=False); print time.time()-t1




