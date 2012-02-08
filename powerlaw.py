def distribution_fit(data, distribution, discrete=False, xmin=None, xmax=None, find_xmin=False, find_xmax=False, comparison_alpha=None, force_positive_mean=False):
    """distribution_fit does things"""
    from numpy import log
    if xmin:
        xmin = float(xmin)
        data = data[data>=xmin]
    if xmax:
        xmax = float(xmax)
        data = data[data<=xmax]

    if len(data)<2:
        from numpy import array
        from sys import float_info
        parameters = array([0, 0])
        loglikelihood = -10**float_info.max_10_exp
        if comparison_alpha==None:
            return parameters, loglikelihood
        R = 10**float_info.max_10_exp
        p = 1
        return parameters, loglikelihood, R, p

    n = float(len(data))

    if distribution=='power_law' and not discrete and xmin and not xmax:
        from numpy import array, nan
        alpha = 1+n/\
                sum(log(data/xmin))
        loglikelihood = n*log(alpha-1.0) - n*log(xmin) - alpha*sum(log(data/xmin))
        if loglikelihood == nan:
            loglikelihood=0

        parameters = array([alpha])
        return parameters, loglikelihood
    
    else:
        if distribution=='power_law':
            if discrete:
                #initial_parameters=[1 + n/sum( log( data/(xmin-.5) ))]
                initial_parameters=[1 + n/sum( log( data/(xmin) ))]
            else:
                initial_parameters=[1 + n/sum( log( data/xmin ))]
            likelihood_function = lambda parameters:\
                    power_law_likelihoods(\
                    data, parameters[0], xmin, xmax, discrete)

        elif distribution=='exponential':
            from numpy import mean
            initial_parameters=[1/mean(data)]
            likelihood_function = lambda parameters:\
                    exponential_likelihoods(\
                    data, parameters[0], xmin, xmax, discrete)

        elif distribution=='truncated_power_law':
            from numpy import mean
            if discrete:
                #initial_parameters=[1 + n/sum( log( data/(xmin-.5) )), 1/mean(data)]
                initial_parameters=[1 + n/sum( log( data/(xmin) )), 1/mean(data)]
            else:
                initial_parameters=[1 + n/sum( log( data/xmin )), 1/mean(data)]
            likelihood_function = lambda parameters:\
                    truncated_power_law_likelihoods(\
                    data, parameters[0], parameters[1], xmin, xmax, discrete)

        elif distribution=='lognormal':
            from numpy import mean, std
            logdata = log(data)
            initial_parameters=[mean(logdata), std(logdata)]
            likelihood_function = lambda parameters:\
                    lognormal_likelihoods(\
                    data, parameters[0], parameters[1], xmin, xmax, discrete, force_positive_mean=force_positive_mean)

        from scipy.optimize import fmin
        parameters, negative_loglikelihood, iter, funcalls, warnflag, = \
                fmin(\
                lambda p: -sum(log(likelihood_function(p))),\
                initial_parameters, full_output=1, disp=False)
        loglikelihood =-negative_loglikelihood

    if comparison_alpha==None:
        return parameters, loglikelihood

    pl_likelihoods = power_law_likelihoods(data, alpha=comparison_alpha, xmin=xmin, xmax=xmax, discrete=discrete)
    candidate_likelihoods = likelihood_function(parameters)
    R, p = loglikelihood_ratio(pl_likelihoods, candidate_likelihoods)
    return parameters, loglikelihood, R, p

def loglikelihood_ratio(likelihoods1, likelihoods2):
    from numpy import sqrt,log
    from scipy.special import erfc

    n = float(len(likelihoods1))

    loglikelihoods1 = log(likelihoods1)
    loglikelihoods2 = log(likelihoods2)

    R = sum(loglikelihoods1-loglikelihoods2)

    sigma = sqrt( \
    sum(\
    ( (loglikelihoods1-loglikelihoods2) - \
    (loglikelihoods1.mean()-loglikelihoods2.mean()) )**2 \
    )/n )

    p = erfc( abs(R) / (sqrt(2*n)*sigma) ) 
    return R, p

def find_xmin(data, discrete=False, xmax=None):
    from numpy import sort, unique, asarray, argmin, hstack, arange, sqrt
    if xmax:
        data = data[data<=xmax]
    noise_flag=False
#Much of the rest of this function was inspired by Adam Ginsburg's plfit code, specifically around lines 131-143 of this version: http://code.google.com/p/agpy/source/browse/trunk/plfit/plfit.py?spec=svn359&r=357
    if not all(data[i] <= data[i+1] for i in xrange(len(data)-1)):
        data = sort(data)
    xmins, xmin_indices = unique(data, return_index=True)
    xmins = xmins[:-1]
    if len(xmins)<2:
        from sys import float_info
        xmin = 1
        D = 1
        alpha = 0
        loglikelihood = -10**float_info.max_10_exp
        n_tail = 1
        noise_flag = True
        return xmin, D, alpha, loglikelihood, n_tail, noise_flag
    xmin_indices = xmin_indices[:-1] #Don't look at last xmin, as that's also the xmax, and we want to at least have TWO points to fit!

    alpha_MLE_function = lambda xmin: distribution_fit(data, 'power_law', xmin=xmin, xmax=xmax, discrete=discrete)
    fits  = asarray( map(alpha_MLE_function,xmins))
    #import pdb; pdb.set_trace();
    alphas = hstack(fits[:,0])
    loglikelihoods = fits[:,1]

    ks_function = lambda index: power_law_ks_distance(data, alphas[index], xmins[index], xmax=xmax, discrete=discrete)
    Ds  = asarray( map(ks_function, arange(len(xmins))))

    sigmas = (alphas-1)/sqrt(len(data)-xmin_indices+1)
    good_values = sigmas<.1
    xmin_max = argmin(good_values)
    if xmin_max>0 and not good_values[-1]==True:
        Ds = Ds[:xmin_max]
        alphas = alphas[:xmin_max]
    else:
        noise_flag = True

    min_D_index = argmin(Ds)
    xmin = xmins[argmin(Ds)]
    D = Ds[min_D_index]
    alpha = alphas[min_D_index]
    loglikelihood = loglikelihoods[min_D_index]
    n_tail = sum(data>=xmin)

    return xmin, D, alpha, loglikelihood, n_tail, noise_flag

def power_law_ks_distance(data, alpha, xmin, xmax=None, discrete=False, kuiper=False):
    """Data must be sorted beforehand!"""
    from numpy import arange, sort, mean
    data = data[data>=xmin]
    if xmax:
        data = data[data<=xmax]
    n = float(len(data))
    if n<2:
        if kuiper:
            return 1, 1, 2
        return 1

    if not all(data[i] <= data[i+1] for i in arange(n-1)):
        data = sort(data)

    if not discrete:
        Actual_CDF = arange(n)/n
        Theoretical_CDF = 1-(data/xmin)**(-alpha+1)

    if discrete:
        from scipy.special import zeta
        if xmax:
            Actual_CDF, bins = cumulative_distribution_function(data,xmin=xmin,xmax=xmax)
            Theoretical_CDF = 1 - ((zeta(alpha, bins) - zeta(alpha, xmax+1)) /\
                    (zeta(alpha, xmin)-zeta(alpha,xmax+1)))
        if not xmax:
            Actual_CDF, bins = cumulative_distribution_function(data,xmin=xmin)
            Theoretical_CDF = 1 - (zeta(alpha, bins) /\
                    zeta(alpha, xmin))

    D_plus = max(Theoretical_CDF-Actual_CDF)
    D_minus = max(Actual_CDF-Theoretical_CDF)
    Kappa = 1 + mean(Theoretical_CDF-Actual_CDF)

    if kuiper:
        return D_plus, D_minus, Kappa

    D = max(D_plus, D_minus)

    return D

def cumulative_distribution_function(data, xmin=None, xmax=None, survival=False):
    from numpy import cumsum, histogram, arange, array
    if type(data)==list:
        data = array(data)
    if not data.any():
        return array([0]), array([0])
    if xmin:
        data = data[data>=xmin]
    else:
        xmin = min(data)
    if xmax:
        data = data[data<=xmax]
    else:
        xmax = max(data)
    
    if discrete(data):
        CDF = cumsum(histogram(data,arange(xmin-1, xmax+2), density=True)[0])[:-1]
        bins = arange(xmin, xmax+1)
    else:
        n = float(len(data))
        CDF = arange(n)/n
        if not all(data[i] <= data[i+1] for i in arange(n-1)):
            from numpy import sort
            data = sort(data)
        bins = data

    if survival:
        CDF = 1-CDF
    return CDF, bins

def discrete(data):
    from numpy import floor
    return reduce(lambda x,y: x==True and floor(y)==float(y), data,True)

def power_law_likelihoods(data, alpha, xmin, xmax=False, discrete=False):
    if alpha<0:
        from numpy import tile
        from sys import float_info
        return tile(10**float_info.min_10_exp, len(data))

    data = data[data>=xmin]
    if xmax:
        data = data[data<=xmax]

    if not discrete:
        likelihoods = (data**-alpha)/\
                ( (alpha-1) * xmin**(alpha-1) )
    if discrete:
        if alpha<1:
            from numpy import tile
            from sys import float_info
            return tile(10**float_info.min_10_exp, len(data))
        if not xmax:
            from scipy.special import zeta
            likelihoods = (data**-alpha)/\
                    zeta(alpha, xmin)
        if xmax:
            from scipy.special import zeta
            likelihoods = (data**-alpha)/\
                    (zeta(alpha, xmin)-zeta(alpha,xmax+1))
    from sys import float_info
    likelihoods[likelihoods==0] = 10**float_info.min_10_exp
    return likelihoods

def exponential_likelihoods(data, gamma, xmin, xmax=False, discrete=False):
    if gamma<0:
        from numpy import tile
        from sys import float_info
        return tile(10**float_info.min_10_exp, len(data))

    data = data[data>=xmin]
    if xmax:
        data = data[data<=xmax]

    from numpy import exp
    if not discrete:
#        likelihoods = exp(-gamma*data)*\
#                gamma*exp(gamma*xmin)
        likelihoods = gamma*exp(gamma*(xmin-data)) #Simplified so as not to throw a nan from infs being divided by each other
    if discrete:
        if not xmax:
            likelihoods = exp(-gamma*data)*\
                    (1-exp(-gamma))*exp(gamma*xmin)
        if xmax:
            likelihoods = exp(-gamma*data)*\
                    (1-exp(-gamma))/(exp(-gamma*xmin)-exp(-gamma*(xmax+1)))
    from sys import float_info
    likelihoods[likelihoods==0] = 10**float_info.min_10_exp
    return likelihoods

def truncated_power_law_likelihoods(data, alpha, gamma, xmin, xmax=False, discrete=False):
    if alpha<0 or gamma<0:
        from numpy import tile
        from sys import float_info
        return tile(10**float_info.min_10_exp, len(data))

    data = data[data>=xmin]
    if xmax:
        data = data[data<=xmax]

    from numpy import exp
    if not discrete:
        from mpmath import gammainc
#        likelihoods = (data**-alpha)*exp(-gamma*data)*\
#                (gamma**(1-alpha))/\
#                float(gammainc(1-alpha,gamma*xmin))
        likelihoods = (gamma**(1-alpha))/\
                ( (data**alpha) * exp(gamma*data) * gammainc(1-alpha,gamma*xmin) ).astype(float) #Simplified so as not to throw a nan from infs being divided by each other
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange
            X = arange(xmin, xmax+1)
            PDF = (X**-alpha)*exp(-gamma*X)
            PDF = PDF/sum(PDF)
            likelihoods = PDF[(data-xmin).astype(int)]
    from sys import float_info
    likelihoods[likelihoods==0] = 10**float_info.min_10_exp
    return likelihoods

def lognormal_likelihoods(data, mu, sigma, xmin, xmax=False, discrete=False, force_positive_mean=False):
    from numpy import log
    if sigma<=0 or mu<log(xmin): 
        #The standard deviation can't be negative, and the mean of the logarithm of the distribution can't be smaller than the log of the smallest member of the distribution!
        from numpy import tile
        from sys import float_info
        return tile(10**float_info.min_10_exp, len(data))

    if force_positive_mean and mu<=0:
        from numpy import tile
        from sys import float_info
        return tile(10**float_info.min_10_exp, len(data))

    data = data[data>=xmin]
    if xmax:
        data = data[data<=xmax]

    if not discrete:
        from numpy import sqrt, exp
#        from mpmath import erfc
        from scipy.special import erfc
        from scipy.constants import pi
        likelihoods = (1.0/data)*exp(-( (log(data) - mu)**2 )/2*sigma**2)*\
                sqrt(2/(pi*sigma**2))/erfc( (log(xmin)-mu) / (sqrt(2)*sigma))
#        likelihoods = likelihoods.astype(float)
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange, exp
#            from mpmath import exp
            X = arange(xmin, xmax+1)
#            PDF_function = lambda x: (1.0/x)*exp(-( (log(x) - mu)**2 ) / 2*sigma**2)
#            PDF = asarray(map(PDF_function,X))
            PDF = (1.0/X)*exp(-( (log(X) - mu)**2 ) / 2*sigma**2)
            PDF = (PDF/sum(PDF)).astype(float)
            likelihoods = PDF[(data-xmin).astype(int)]
    from sys import float_info
    likelihoods[likelihoods==0] = 10**float_info.min_10_exp
    return likelihoods

def hist_log(data, max_size=False, min_size=False, plot=True, show=True):
    """hist_log does things"""
    from numpy import logspace, histogram
    from math import ceil, log10
    import pylab
    if ~max_size:
        max_size = max(data)
    if ~min_size:
        min_size = min(data)
    log_min_size = log10(min_size)
    log_max_size = log10(max_size)
    number_of_bins = ceil((log_max_size-log_min_size)*10)
    bins=logspace(log_min_size, log_max_size, num=number_of_bins)
    hist, edges = histogram(data, bins, density=True)
    if plot:
        pylab.plot(edges[:-1], hist, 'o')
        pylab.gca().set_xscale("log")
        pylab.gca().set_yscale("log")
        if show:
            pylab.show()
    return (hist, edges)

def plot_cdf(variable, name=None, x_label=None, xmin=None, xmax=None, survival=True):
    import matplotlib.pyplot as plt
    iCDF, bins = cumulative_distribution_function(variable, survival=survival, xmin=xmin, xmax=xmax)
    plt.plot(bins, iCDF)
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    if name and survival:
        plt.title(name +' survival function')
    elif name:
        plt.title(name +' CDF')

    plt.xlabel(x_label)
    if survival:
        plt.ylabel('P(X>x)')
    else:
        plt.ylabel('P(X<x)')
    return (iCDF, bins)
