# intended to implement a power-law fitting routine as specified in.....
# that paper Erik sent me in July 2009.  
# http://www.santafe.edu/~aaronc/powerlaws/
#
# The MLE for the power-law alpha is very easy to derive given knowledge
# of the lowest value at which a power law holds, but that point is 
# difficult to derive and must be acquired iteratively.
from pylab import *

# def plfit(x,xmin=None):
#   if xmin is None:
#     raise InputError("can't do that yet")
#     xguess = mean(x)
#     mle = 1+len(x) / (sum(log(x[x>xguess]/xguess)))
# 
#   return 1+len(x) / (sum(log(x[x>xmin]/xmin)))

def plfit_lsq(x,y):
    """
    Returns A and B in y=Ax^B
    http://mathworld.wolfram.com/LeastSquaresFittingPowerLaw.html
    """
    n = len(x)
    btop = n * (log(x)*log(y)).sum() - (log(x)).sum()*(log(y)).sum()
    bbottom = n*(log(x)**2).sum() - (log(x).sum())**2
    b = btop / bbottom
    a = ( log(y).sum() - b * log(x).sum() ) / n

    A = exp(a)
    return A,b


def plfit(x,nosmall=False,finite=False):
    """
    A Python implementation of the Matlab code http://www.santafe.edu/~aaronc/powerlaws/plfit.m
    from http://www.santafe.edu/~aaronc/powerlaws/

    See A. Clauset, C.R. Shalizi, and M.E.J. Newman, "Power-law distributions
    in empirical data" SIAM Review, to appear (2009). (arXiv:0706.1062)
    http://arxiv.org/abs/0706.1062
    """
    xmins = unique(x)
    xmins = xmins[1:-1]
    dat = xmins * 0 
    z = sort(x)
    for xm in arange(len(xmins)):
        xmin = xmins[xm]
        z    = z[z>=xmin] 
        n    = float(len(z))
        # estimate alpha using direct MLE
        a    =  n / sum( log(z/xmin) )
        if nosmall:
            # 4. For continuous data, PLFIT can return erroneously large estimates of 
            #    alpha when xmin is so large that the number of obs x >= xmin is very 
            #    small. To prevent this, we can truncate the search over xmin values 
            #    before the finite-size bias becomes significant by calling PLFIT as
            if (a-1)/sqrt(n) > 0.1:
                #dat(xm:end) = [];
                dat = dat[:xm]
                xm = len(xmins)+1
                break
        # compute KS statistic
        cx   = arange(n)/float(n)  #data
        cf   = 1-(xmin/z)**a  # fitted
        dat[xm] = max( abs(cf-cx) )
    D     = min(dat);
    #xmin  = xmins(find(dat<=D,1,'first'));
    xmin  = xmins[argmin(dat)]
    z     = x[x>=xmin]
    n     = len(z)
    alpha = 1 + n / sum( log(z/xmin) )
    if finite:
        alpha = alpha*(n-1)/n+1/n
    if n < 50 and ~finite:
        print '(PLFIT) Warning: finite-size bias may be present.'
    L = n*log((alpha-1)/xmin) - alpha*sum(log(z/xmin));
    return xmin,alpha,L,dat

def plotcdf(x,xmin,alpha):
    """
    Plots CDF and powerlaw
    """

    x=sort(x)
    n=len(x)
    xcdf = arange(n,0,-1,dtype='float')/float(n)

    q = x[x>=xmin]
    fcdf = (q/xmin)**(1-alpha)
    nc = xcdf[argmax(x>=xmin)]
    fcdf_norm = nc*fcdf

    loglog(x,xcdf)
    loglog(q,fcdf_norm)

def plotpdf(x,xmin,alpha,nbins=30,dolog=False):
    """
    Plots PDF and powerlaw....
    """

    x=sort(x)
    n=len(x)

    if dolog:
        hb = hist(x,bins=logspace(log10(min(x)),log10(max(x)),nbins),log=True)
        alpha += 1
    else:
        hb = hist(x,bins=linspace((min(x)),(max(x)),nbins))
    h,b=hb[0],hb[1]
    b = b[1:]

    q = x[x>=xmin]
    px = (alpha-1)/xmin * (q/xmin)**(-alpha)

    arg = argmin(abs(b-xmin))
    norm = mean( h[b>xmin] / ((alpha-1)/xmin * (b[b>xmin]/xmin)**(-alpha))  )
    px = px*norm

    loglog(q,px)

    gca().set_xlim(min(x),max(x))

