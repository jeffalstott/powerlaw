class cache(object):    
    '''Computes attribute value and caches it in the instance.
    Python Cookbook (Denis Otkidach) http://stackoverflow.com/users/168352/denis-otkidach
    This decorator allows you to create a property which can be computed once and
    accessed many times. Sort of like memoization.
    '''
    def __init__(self, method, name=None):
        # record the unbound-method and the name
        self.method = method
        self.name = name or method.__name__
        self.__doc__ = method.__doc__
    def __get__(self, inst, cls):
        # self: <__main__.cache object at 0xb781340c>
        # inst: <__main__.Foo object at 0xb781348c>
        # cls: <class '__main__.Foo'>       
        if inst is None:
            # instance attribute accessed on class, return self
            # You get here if you write `Foo.bar`
            return self
        # compute, cache and return the instance's attribute value
        result = self.method(inst)
        # setattr redefines the instance's attribute so this doesn't get called again
        setattr(inst, self.name, result)
        return result


class Fit(object):
    def __init__(self, data,
        discrete=False,
        xmin=None, xmax=None,
        fit_method='Likelihood',
        estimate_discrete=True,
        discrete_approximation='round',
        sigma_threshold=None):

        self.data_original = data
        from numpy import asarray
        self.data = asarray(self.data_original)

        self.discrete = discrete

        self.fit_method = fit_method
        self.estimate_discrete = estimate_discrete
        self.discrete_approximation = discrete_approximation
        self.sigma_threshold = sigma_threshold

        self.given_xmin = xmin
        self.given_xmax = xmax
        self.xmin = self.given_xmin
        self.xmax = self.given_xmax

        if 0 in self.data:
            print("Value 0 in data. Throwing out 0 values")
            self.data = self.data[self.data!=0]

        if self.xmax:
            self.fixed_xmax = True
            n_above_max = sum(self.data>self.xmax)
            self.data = self.data[self.data<=self.xmax]
        else:
            n_above_max = 0
            self.fixed_xmax = False

        if not all(self.data[i] <= self.data[i+1] for i in xrange(len(self.data)-1)):
            from numpy import sort
            self.data = sort(self.data)

        if xmin and type(xmin)!=tuple and type(xmin)!=list:
            self.fixed_xmin = True
            self.xmin = xmin
            self.noise_flag = None
            pl = Power_Law(xmin = self.xmin,
                xmax = self.xmax,
                discrete = self.discrete,
                estimate_discrete = self.estimate_discrete,
                data = self.data)
            self.D = pl.D
            self.alpha = pl.alpha
            self.sigma = pl.sigma
            self.loglikelihood = pl.loglikelihood
            self.power_law = pl
        else:
            self.fixed_xmin=False
            print("Calculating best minimal value for power law fit")
            self.find_xmin()

        self.data = self.data[self.data>=self.xmin]
        self.n = float(len(self.data))
        self.n_tail = self.n + n_above_max

        self.supported_distributions = ['power_law',
            'lognormal',
            'exponential',
            'truncated_power_law',
            'stretched_exponential',
            'gamma']

    def __getattr__(self, name):
        if name in self.supported_distributions:
            from string import capwords
            dist = capwords(name, '_')
            dist = globals()[dist] #Seems a hack. Might try import powerlaw; getattr(powerlaw, dist)
            setattr(self, name,
                dist(data=self.data,
                xmin=self.xmin, xmax=self.xmax,
                discrete=self.discrete,
                fit_method=self.fit_method,
                estimate_discrete=self.estimate_discrete,
                discrete_approximation=self.discrete_approximation))
            return getattr(self, name)
        else:  raise AttributeError, name

    def find_xmin(self):
        from numpy import unique, asarray, argmin
#Much of the rest of this function was inspired by Adam Ginsburg's plfit code,
#specifically the mapping and sigma threshold behavior:
#http://code.google.com/p/agpy/source/browse/trunk/plfit/plfit.py?spec=svn359&r=357
        if not self.given_xmin:
            possible_xmins = self.data
        else:
            possible_xmins = self.data[
                (min(self.given_xmin)<=self.data)<=max(self.given_xmin)]
        xmins, xmin_indices = unique(possible_xmins, return_index=True)
#Don't look at last xmin, as that's also the xmax, and we want to at least have TWO points to fit!
        xmins = xmins[:-1]
        xmin_indices = xmin_indices[:-1] 
        if len(xmins)<=0:
            print("Less than 2 unique data values left after xmin and xmax options!"
            "Cannot fit.")
            from sys import float_info
            self.xmin = 1
            self.D = 1
            self.alpha = 0
            self.sigma = 1
            self.loglikelihood = -10**float_info.max_10_exp
            self.n_tail = 0
            self.Ds = 1
            self.alphas = 0
            self.sigmas = 1

        def fit_function(xmin):
            pl = Power_Law(xmin = xmin,
                xmax = self.xmax,
                discrete = self.discrete,
                estimate_discrete = self.estimate_discrete,
                data = self.data)
            return pl.D, pl.alpha, pl.loglikelihood, pl.sigma

        fits  = asarray( map(fit_function, xmins))
        self.Ds = fits[:,0]
        self.alphas = fits[:,1]
        self.loglikelihoods = fits[:,2]
        self.sigmas = fits[:,3]
        self.xmins = xmins

        if self.sigma_threshold:
            good_values = self.sigmas < self.sigma_threshold
            #Find the last good value (The first False, where sigma > threshold)
            xmin_max = argmin(good_values)
            #If there are no fits beyond the noise threshold
            if good_values.all():
                min_D_index = argmin(self.Ds)
                self.noise_flag = False
            elif xmin_max>0:
                min_D_index = argmin(self.Ds[:xmin_max])
                self.noise_flag = False
            else:
                min_D_index = argmin(self.Ds)
                self.noise_flag = True
        else:
            min_D_index = argmin(self.Ds)
            self.noise_flag = False

        self.xmin = xmins[min_D_index]
        self.D = self.Ds[min_D_index]
        self.alpha = self.alphas[min_D_index]
        self.sigma = self.sigmas[min_D_index]
        self.loglikelihood = self.loglikelihoods[min_D_index]

        return self.xmin


    def nested_distribution_compare(self, dist1, dist2, **kwargs):
        return self.distribution_compare(dist1, dist2, nested=True, **kwargs)

    def distribution_compare(self, dist1, dist2, nested=None, **kwargs):
        if (dist1 in dist2) or (dist2 in dist1) and nested==None:
            print "Assuming nested distributions"
            nested = True

        dist1 = getattr(self, dist1)
        dist2 = getattr(self, dist2)

        loglikelihoods1 = dist1.loglikelihoods(self.data)
        loglikelihoods2 = dist2.loglikelihoods(self.data)

        return loglikelihood_ratio(
            loglikelihoods1, loglikelihoods2,
            nested=nested,
            **kwargs)

    def loglikelihood_ratio(self, dist1, dist2, nested=None, **kwargs):
        return self.distribution_compare(dist1, dist2, nested=nested, **kwargs)

    def cdf(self, original_data=False, survival=False, **kwargs):
        if original_data:
            data = self.data_original
            xmin = None
            xmax = None
        else:
            data = self.data
            xmin = self.xmin
            xmax = self.xmax
        return cdf(data, xmin=xmin, xmax=xmax, survival=survival,
                **kwargs) 

    def ccdf(self, original_data=False, survival=True, **kwargs):
        if original_data:
            data = self.data_original
            xmin = None
            xmax = None
        else:
            data = self.data
            xmin = self.xmin
            xmax = self.xmax
        return cdf(data, xmin=xmin, xmax=xmax, survival=survival,
                **kwargs) 

    def pdf(self, original_data=False, **kwargs):
        if original_data:
            data = self.data_original
            xmin = None
            xmax = None
        else:
            data = self.data
            xmin = self.xmin
            xmax = self.xmax
        edges, hist = pdf(data, xmin=xmin, xmax=xmax, **kwargs)
        return edges, hist

    def plot_cdf(self, ax=None, original_data=False, survival=False, **kwargs):
        if original_data:
            data = self.data_original
        else:
            data = self.data
        return plot_cdf(data, ax=ax, survival=survival, **kwargs)

    def plot_ccdf(self, ax=None, original_data=False, survival=True, **kwargs):
        if original_data:
            data = self.data_original
        else:
            data = self.data
        return plot_cdf(data, ax=ax, original_data=original_data,
                survival=survival, **kwargs)

    def plot_pdf(self, ax=None, original_data=False, **kwargs):
        if original_data:
            data = self.data_original
        else:
            data = self.data
        return plot_pdf(data, ax=ax, **kwargs)

class Distribution(object):

    def __init__(self,
        xmin=1, xmax=None,
        discrete=False,
        fit_method='Likelihood',
        data = None,
        parameters = None,
        invalid_parameters = None,
        discrete_approximation = 'round',
        **kwargs):

        self.xmin = xmin
        self.xmax = xmax
        self.discrete = discrete
        self.fit_method = fit_method
        self.discrete_approximation = discrete_approximation

        self.parameter1 = None
        self.parameter2 = None
        self.parameter3 = None
        self.parameter1_name = None
        self.parameter2_name = None
        self.parameter3_name = None

        if parameters!=None:
            self.parameters(parameters)

        if invalid_parameters!=None:
            self.invalid_parameters(invalid_parameters)
        else:
            self._invalid_parameters = None

        if data!=None:
            self.fit(data)

    def fit(self, data):
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        if self.fit_method=='Likelihood':
            def fit_function(params):
                self.parameters(params)
                return -sum(self.loglikelihoods(data))
        elif self.fit_method=='KS':
            def fit_function(params):
                self.parameters(params)
                self.KS(data)
                return self.D
        from scipy.optimize import fmin
        parameters, negative_loglikelihood, iter, funcalls, warnflag, = \
            fmin(
                lambda params: fit_function(params),
                self.initial_parameters(data),
                full_output=1,
                disp=False)
        self.parameters(parameters)
        self.loglikelihood =-negative_loglikelihood
        self.KS(data)

    def KS(self, data):
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        if len(data)<2:
            self.D = 1
            self.D_plus = 1
            self.D_minus = 1
            self.Kappa = 2

        bins, Actual_CDF = cdf(data)
        Theoretical_CDF = self.cdf(bins)

        self.D_plus = max(Theoretical_CDF-Actual_CDF)
        self.D_minus = max(Actual_CDF-Theoretical_CDF)
        from numpy import mean
        self.Kappa = 1 + mean(Theoretical_CDF-Actual_CDF)

        self.V = self.D_plus + self.D_minus
        self.D = max(self.D_plus, self.D_minus)

        return self.D

    def ccdf(self,x, survival=True):
        return self.cdf(x, survival=survival)

    def cdf(self,x, survival=False):
        x = trim_to_range(x, xmin=self.xmin, xmax=self.xmax)
        n = len(x)
        from sys import float_info
        if self._invalid_parameters:
            from numpy import tile
            return tile(10**float_info.min_10_exp, n)

        CDF = self._cdf_base_function(x) - self._cdf_base_function(self.xmin)
        if self.xmax:
            CDF = CDF - self._cdf_base_function(self.xmax)

        norm = 1 - self._cdf_base_function(self.xmin)
        if self.xmax:
            norm = norm - (1 - self._cdf_base_function(self.xmax))

        CDF = CDF/norm

        if survival:
            CDF = 1 - CDF

        return CDF

    def pdf(self, data):
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        n = len(data)
        from sys import float_info
        if self._invalid_parameters:
            from numpy import tile
            return tile(10**float_info.min_10_exp, n)

        if not self.discrete:
            f = self._pdf_base_function(data)
            C = self._pdf_continuous_normalizer
            likelihoods = f*C
        else:
            if self._pdf_discrete_normalizer:
                f = self._pdf_base_function(data)
                C = self._pdf_discrete_normalizer
                likelihoods = f*C
            elif self.discrete_approximation=='round':
                lower_data = data-.5
                upper_data = data+.5
#Temporarily expand xmin and xmax to be able to grab the extra bit of
#probability mass beyond the (integer) values of xmin and xmax
#Note this is a design decision. One could also say this extra 
#probability "off the edge" of the distribution shouldn't be included,
#and that implementation is retained below, commented out. Note, however,
#that such a cliff means values right at xmin and xmax have half the width to
#grab probability from, and thus are lower probability than they would otherwise
#be. This is particularly concerning for values at xmin, which are typically 
#the most likely and greatly influence the distribution's fit.
                self.xmin -= .5
                if self.xmax:
                    self.xmax -= .5
                #Clean data for invalid values before handing to cdf, which will purge them
                #lower_data[lower_data<self.xmin] +=.5
                #if self.xmax:
                #    upper_data[upper_data>self.xmax] -=.5
                likelihoods = self.cdf(upper_data)-self.cdf(lower_data)
                self.xmin +=.5
                if self.xmax:
                    self.xmax += .5
            else:
                if self.discrete_approximation=='xmax':
                    upper_limit = self.xmax
                else:
                    upper_limit = self.discrete_approximation
#            from mpmath import exp
                    from numpy import arange
                    X = arange(self.xmin, upper_limit+1)
                    PDF = self._pdf_base_function(X)
                    PDF = (PDF/sum(PDF)).astype(float)
                    likelihoods = PDF[(data-self.xmin).astype(int)]
        likelihoods[likelihoods==0] = 10**float_info.min_10_exp
        return likelihoods

    def invalid_parameters(self, function):
        if function=='basic_assumptions':
            self._invalid_parameters = self._basic_assumptions
        else:
            self._invalid_parameters = function

    def likelihoods(self, data):
        return self.pdf(data) 

    def loglikelihoods(self, data):
        from numpy import log
        return log(self.likelihoods(data))

    def plot_ccdf(self, data, ax=None, survival=True, **kwargs):
        return self.plot_cdf(data, ax=None, survival=survival, **kwargs)

    def plot_cdf(self, data, ax=None, survival=False, **kwargs):
        from numpy import unique
        bins = unique(data)
        CDF = self.cdf(bins, survival=survival)
        if not ax:
            import matplotlib.pyplot as plt
            plt.plot(bins, CDF, **kwargs)
            ax = plt.gca()
        else:
            ax.plot(bins, CDF, **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        return ax

    def plot_pdf(self, data, ax=None, **kwargs):
        from numpy import unique
        bins = unique(data)
        PDF = self.pdf(bins)
        if not ax:
            import matplotlib.pyplot as plt
            plt.plot(bins, PDF, **kwargs)
            ax = plt.gca()
        else:
            ax.plot(bins, PDF, **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        return ax

class Power_Law(Distribution):

    def __init__(self, estimate_discrete=True, **kwargs):
        self.estimate_discrete = estimate_discrete
        Distribution.__init__(self, **kwargs)

    def parameters(self, params):
        self.alpha = params[0]
        self.parameter1 = self.alpha
        self.parameter1_name = 'alpha'

    @property
    def name(self):
        return "power_law"

    def fit(self, data):
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        n = len(data)
        from numpy import log, nan
        if not self.discrete and not self.xmax:
            self.alpha = 1 + ( n / sum( log( data / self.xmin ) ))
            self.loglikelihood = ( n*log(self.alpha-1.0) -
                n*log(self.xmin) -
                self.alpha*sum(log(data/self.xmin)) )
            if self.loglikelihood == nan:
                self.loglikelihood=0
            self.KS(data)
        elif self.discrete and self.estimate_discrete and not self.xmax:
            self.alpha = 1 + ( n / sum( log( data / ( self.xmin - .5 ) ) ))
            self.loglikelihood = ( n*log(self.alpha-1.0) -
                n*log(self.xmin) -
                self.alpha*sum(log(data/self.xmin)) )
            if self.loglikelihood == nan:
                self.loglikelihood=0
            self.KS(data)
        else:
            Distribution.fit(self, data)
        from numpy import sqrt
        self.sigma = (self.alpha - 1) / sqrt(n)

    def initial_parameters(self, data):
        from numpy import log, sum
        return 1 + len(data)/sum( log( data / (self.xmin) ))

    def cdf(self,x, survival=False):
        if not self.discrete:
            CDF = 1-(x/self.xmin)**(-self.alpha+1)
        else:
            from scipy.special import zeta
            if self.xmax:
                CDF= 1 - ((zeta(self.alpha, x) -
                            zeta(self.alpha, self.xmax+1)) /
                            (zeta(self.alpha, self.xmin) -
                            zeta(self.alpha,self.xmax+1)))
            else:
                CDF = 1 - (zeta(self.alpha, x) /  zeta(self.alpha, self.xmin))
        if survival:
            CDF = 1 - CDF
        return CDF

    def pdf(self, data):
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        n = len(data)
        from sys import float_info

        if self.alpha<0:
            from numpy import tile
            return tile(10**float_info.min_10_exp, n)

        f = data**-self.alpha

        if not self.discrete:
            C = (self.alpha-1) * self.xmin**(self.alpha-1)
        else:
            if self.alpha<1:
                from numpy import tile
                from sys import float_info
                return tile(10**float_info.min_10_exp, n)
            if not self.xmax:
                from scipy.special import zeta
                C = 1 / zeta(self.alpha, self.xmin)
            if self.xmax:
                from scipy.special import zeta
                C = 1 / (zeta(self.alpha, self.xmin) -
                        zeta(self.alpha,self.xmax+1))

        likelihoods = f*C

        likelihoods[likelihoods==0] = 10**float_info.min_10_exp
        return likelihoods

class Exponential(Distribution):

    def parameters(self, params):
        self.Lambda = params[0]
        self.parameter1 = self.Lambda
        self.parameter1_name = 'lambda'

    @property
    def name(self):
        return "exponential"

    def initial_parameters(self, data):
        from numpy import mean
        return 1/mean(data)

    def cdf(self,x):
        x = trim_to_range(x, xmin=self.xmin, xmax=self.xmax)
        n = len(x)
        from sys import float_info
        if self.Lambda<0:
            from numpy import tile
            return tile(10**float_info.min_10_exp, n)

        from numpy import exp
        CDF = exp(-self.Lambda*x)
        if self.xmax:
            CDF = CDF - exp( -self.Lambda*( self.xmax+1 ))

        norm = exp(-self.Lambda*self.xmin)
        if self.xmax:
            norm = norm - exp( -self.Lambda*( self.xmax+1 ))

        CDF = 1 - (CDF/norm)

        return CDF
    

    def pdf(self, data):
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        n = len(data)
        from sys import float_info
        if self.Lambda<0:
            from numpy import tile
            return tile(10**float_info.min_10_exp, n)

        from numpy import exp
        f = exp(-self.Lambda*data)

        if not self.discrete:
#        likelihoods = exp(-Lambda*data)*\
#                Lambda*exp(Lambda*xmin)
            likelihoods = self.Lambda*exp(self.Lambda*(self.xmin-data))
            #Simplified so as not to throw a nan from infs being divided by each other
        if self.discrete:
            if not self.xmax:
                C = (1-exp(-self.Lambda))*exp(self.Lambda*self.xmin)
            else:
                C = ( (1-exp(-self.Lambda)) /
                        (exp(-self.Lambda*self.xmin) - 
                            exp(-self.Lambda*(self.xmax+1))
                            )
                        )
            likelihoods = f*C
        likelihoods[likelihoods==0] = 10**float_info.min_10_exp
        return likelihoods

class Lognormal(Distribution):

    def parameters(self, params):
        self.mu = params[0]
        self.parameter1 = self.mu
        self.parameter1_name = 'mu'

        self.sigma = params[1]
        self.parameter2 = self.sigma
        self.parameter2_name = 'sigma'

    @property
    def name(self):
        return "lognormal"

    def initial_parameters(self, data):
        from numpy import mean, std, log
        logdata = log(data)
        return (mean(logdata), std(logdata))

    @property
    def _basic_assumptions(self):
#The standard deviation can't be negative, and the mean of the
#logarithm of the distribution can't be smaller than the log of
#the smallest member of the distribution!
        from numpy import log
        return self.sigma<=0 or self.mu<log(self.xmin)

    def _cdf_base_function(self, x):
        from numpy import sqrt, log
        from scipy.special import erf
        return  0.5 + ( 0.5 * 
                erf((log(x)-self.mu) / (sqrt(2)*self.sigma)))

    def _pdf_base_function(self, x):
        from numpy import exp, log
        return ((1.0/x) *
                exp(-( (log(x) - self.mu)**2 )/(2*self.sigma**2)))

    @property
    def _pdf_continuous_normalizer(self):
#        from mpmath import erfc
        from scipy.special import erfc
        from scipy.constants import pi
        from numpy import sqrt, log
        C = ( sqrt(2/(pi*self.sigma**2)) /
                erfc( (log(self.xmin)-self.mu) / (sqrt(2)*self.sigma))
                )
        return C

    @property
    def _pdf_discrete_normalizer(self):
        return False



def nested_loglikelihood_ratio(loglikelihoods1, loglikelihoods2, **kwargs):
    return loglikelihood_ratio(loglikelihoods1, loglikelihoods2,
            nested=True, **kwargs)

def loglikelihood_ratio(loglikelihoods1, loglikelihoods2,
        nested=False, normalized_ratio=False):
    from numpy import sqrt
    from scipy.special import erfc

    n = float(len(loglikelihoods1))

    if n==0:
        R = 0
        p = 1
        return R, p
    from numpy import asarray
    loglikelihoods1 = asarray(loglikelihoods1)
    loglikelihoods2 = asarray(loglikelihoods2)

    R = sum(loglikelihoods1-loglikelihoods2)

    if nested:
        from scipy.stats import chi2
        p = 1 - chi2.cdf(abs(2*R), 1)
    else:
        from numpy import mean
        mean_diff = mean(loglikelihoods1)-mean(loglikelihoods2)
        variance = sum(
                ( (loglikelihoods1-loglikelihoods2) - mean_diff)**2
                )/n
        p = erfc( abs(R) / sqrt(2*n*variance))

    if normalized_ratio:
        R = R/sqrt(n*variance)

    #import ipdb; ipdb.set_trace()
    return R, p

def cdf(data, survival=False, **kwargs):
    return cumulative_distribution_function(data, survival=survival, **kwargs)

def ccdf(data, survival=True, **kwargs):
    return cumulative_distribution_function(data, survival=survival, **kwargs)

def cumulative_distribution_function(data,
    xmin=None, xmax=None,
    survival=False, **kwargs):

    from numpy import array
    data = array(data)
    if not data.any():
        from numpy import nan
        return array([nan]), array([nan])

    data = trim_to_range(data, xmin=xmin, xmax=xmax)

    n = float(len(data))
    data = checksort(data)

    if is_discrete(data):
        from numpy import searchsorted
        CDF = searchsorted(data, data,side='left')/n
    else:
        from numpy import arange
        CDF = arange(n)/n

    if survival:
        CDF = 1-CDF
    return data, CDF

def is_discrete(data):
    from numpy import floor
    return (floor(data)==data.astype(float)).all()

def trim_to_range(data, xmin=None, xmax=None, **kwargs):
    from numpy import asarray
    data = asarray(data)
    if xmin:
        data = data[data>=xmin]
    if xmax:
        data = data[data<=xmax]
    return data

def pdf(data, xmin=None, xmax=None, linear_bins=False, **kwargs):
    from numpy import logspace, histogram, floor, unique
    from math import ceil, log10
    if not xmax:
        xmax = max(data)
    if not xmin:
        xmin = min(data)
    if linear_bins:
        bins = range(int(xmin), int(xmax))
    else:
        log_min_size = log10(xmin)
        log_max_size = log10(xmax)
        number_of_bins = ceil((log_max_size-log_min_size)*10)
        bins=unique(
                floor(
                    logspace(
                        log_min_size, log_max_size, num=number_of_bins)))
    hist, edges = histogram(data, bins, density=True)
    return edges, hist

def checksort(data):
    """Checks if the data is sorted, in O(n) time. If it isn't sorted, it then"""
    """sorts it in O(nlogn) time. Expectation is that the data will typically"""
    """be sorted."""

    n = len(data)
    from numpy import arange
    if not all(data[i] <= data[i+1] for i in arange(n-1)):
        from numpy import sort
        data = sort(data)
    return data

def plot_ccdf(data, ax=None, survival=False, **kwargs):
    return plot_cdf(data, ax=ax, survival=True, **kwargs)

def plot_cdf(data, ax=None, survival=False, **kwargs):
    bins, CDF = cdf(data, survival=survival, **kwargs)
    if not ax:
        import matplotlib.pyplot as plt
        plt.plot(bins, CDF, **kwargs)
        ax = plt.gca()
    else:
        ax.plot(bins, CDF, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax

def plot_pdf(data, ax=None, **kwargs):
    edges, hist = pdf(data, **kwargs)
    if not ax:
        import matplotlib.pyplot as plt
        plt.plot(edges[:-1], hist, **kwargs)
        ax = plt.gca()
    else:
        ax.plot(edges[:-1], hist, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax

######################
#What follows are functional programming forms of the above code, which are more
#clunky and have somewhat less functionality. However, they are here if your
#really want them.

class Distribution_Fit(object):
    def __init__(self, data, name, xmin, discrete=False, xmax=None, method='Likelihood', estimate_discrete=True):
        self.data = data
        self.discrete = discrete
        self.xmin = xmin
        self.xmax = xmax
        self.method = method
        self.name = name
        self.estimate_discrete = estimate_discrete

        return

    def __getattr__(self, name):
        param_names = {'lognormal': ('mu', 'sigma', None),
                       'exponential': ('Lambda', None, None),
                       'truncated_power_law': ('alpha', 'Lambda', None),
                       'power_law': ('alpha', None, None),
                       'negative_binomial': ('r', 'p', None),
                       'stretched_exponential': ('Lambda', 'beta', None),
                       'gamma': ('k', 'theta', None)}
        param_names = param_names[self.name]

        if name in param_names:
            if name == param_names[0]:
                setattr(self, name, self.parameter1)
            elif name == param_names[1]:
                setattr(self, name, self.parameter2)
            elif name == param_names[2]:
                setattr(self, name, self.parameter3)
            return getattr(self, name)
        elif name in ['parameters',
                      'parameter1_name',
                      'parameter1',
                      'parameter2_name',
                      'parameter2',
                      'parameter3_name',
                      'parameter3',
                      'loglikelihood']:

            self.parameters, self.loglikelihood = distribution_fit(self.data, distribution=self.name, discrete=self.discrete,
                                                                   xmin=self.xmin, xmax=self.xmax, search_method=self.method, estimate_discrete=self.estimate_discrete)
            self.parameter1 = self.parameters[0]
            if len(self.parameters) < 2:
                self.parameter2 = None
            else:
                self.parameter2 = self.parameters[1]
            if len(self.parameters) < 3:
                self.parameter3 = None
            else:
                self.parameter3 = self.parameters[2]

            self.parameter1_name = param_names[0]
            self.parameter2_name = param_names[1]
            self.parameter3_name = param_names[2]

            if name == 'parameters':
                return self.parameters
            elif name == 'parameter1_name':
                return self.parameter1_name
            elif name == 'parameter2_name':
                return self.parameter2_name
            elif name == 'parameter3_name':
                return self.parameter3_name
            elif name == 'parameter1':
                return self.parameter1
            elif name == 'parameter2':
                return self.parameter2
            elif name == 'parameter3':
                return self.parameter3
            elif name == 'loglikelihood':
                return self.loglikelihood
        if name == 'D':
            if self.name != 'power_law':
                self.D = None
            else:
                self.D = power_law_ks_distance(self.data, self.parameter1, xmin=self.xmin, xmax=self.xmax, discrete=self.discrete)
            return self.D
        if name == 'p':
            print("A p value outside of a loglihood ratio comparison to another candidate distribution is not currently supported.\n \
                    If your data set is particularly large and has any noise in it at all, using such statistical tools as the Monte Carlo method\n\
                    can lead to erroneous results anyway; the presence of the noise means the distribution will obviously not perfectly fit the\n\
                    candidate distribution, and the very large number of samples will make the Monte Carlo simulations very close to a perfect\n\
                    fit. As such, such a test will always fail, unless your candidate distribution perfectly describes all elements of the\n\
                    system, including the noise. A more helpful analysis is the comparison between multiple, specific candidate distributions\n\
                    (the loglikelihood ratio test), which tells you which is the best fit of these distributions.")
            self.p = None
            return self.p
#
#        elif name in ['power_law_loglikelihood_ratio',
#                'power_law_p']:
#            pl_R, pl_p = distribution_compare(self.data, 'power_law', self.power_law.parameters, name, self.parameters, self.discrete, self.xmin, self.xmax)
#            self.power_law_loglikelihood_ratio = pl_R
#            self.power_law_p = pl_p
#            if name=='power_law_loglikelihood_ratio':
#                return self.power_law_loglikelihood_ratio
#            if name=='power_law_p':
#                return self.power_law_p
#        elif name in ['truncated_power_law_loglikelihood_ratio',
#                'truncated_power_law_p']:
#            tpl_R, tpl_p = distribution_compare(self.data, 'truncated_power_law', self.truncated_power_law.parameters, name, self.parameters, self.discrete, self.xmin, self.xmax)
#            self.truncated_power_law_loglikelihood_ratio = tpl_R
#            self.truncated_power_law_p = tpl_p
#            if name=='truncated_power_law_loglikelihood_ratio':
#                return self.truncated_power_law_loglikelihood_ratio
#            if name=='truncated_power_law_p':
#                return self.truncated_power_law_p
        else:
            raise AttributeError(name)


def distribution_fit(data, distribution='all', discrete=False, xmin=None, xmax=None, \
        comparison_alpha=None, search_method='Likelihood', estimate_discrete=True):
    """distribution_fit does things"""
    from numpy import log

    if distribution == 'negative_binomial' and not is_discrete(data):
        print("Rounding to integer values for negative binomial fit.")
        from numpy import around
        data = around(data)
        discrete = True

    #If we aren't given an xmin, calculate the best possible one for a power law. This can take awhile!
    if xmin is None or xmin == 'find' or type(xmin) == tuple or type(xmin) == list:
        print("Calculating best minimal value")
        if 0 in data:
            print("Value 0 in data. Throwing out 0 values")
            data = data[data != 0]
        xmin, D, alpha, loglikelihood, n_tail, noise_flag = find_xmin(data, discrete=discrete, xmax=xmax, search_method=search_method, estimate_discrete=estimate_discrete, xmin_range=xmin)
    else:
        alpha = None

    if distribution == 'power_law' and alpha:
        return [alpha], loglikelihood

    xmin = float(xmin)
    data = data[data >= xmin]

    if xmax:
        xmax = float(xmax)
        data = data[data <= xmax]

    #Special case where we call distribution_fit multiple times to do all comparisons
    if distribution == 'all':
        print("Analyzing all distributions")
        print("Calculating power law fit")
        if alpha:
            pl_parameters = [alpha]
        else:
            pl_parameters, loglikelihood = distribution_fit(data, 'power_law', discrete, xmin, xmax, search_method=search_method, estimate_discrete=estimate_discrete)
        results = {}
        results['xmin'] = xmin
        results['xmax'] = xmax
        results['discrete'] = discrete
        results['fits'] = {}
        results['fits']['power_law'] = (pl_parameters, loglikelihood)

        print("Calculating truncated power law fit")
        tpl_parameters, loglikelihood, R, p = distribution_fit(data, 'truncated_power_law', discrete, xmin, xmax, comparison_alpha=pl_parameters[0], search_method=search_method, estimate_discrete=estimate_discrete)
        results['fits']['truncated_power_law'] = (tpl_parameters, loglikelihood)
        results['power_law_comparison'] = {}
        results['power_law_comparison']['truncated_power_law'] = (R, p)
        results['truncated_power_law_comparison'] = {}

        supported_distributions = ['exponential', 'lognormal', 'stretched_exponential', 'gamma']

        for i in supported_distributions:
            print("Calculating %s fit" % i)
            parameters, loglikelihood, R, p = distribution_fit(data, i, discrete, xmin, xmax, comparison_alpha=pl_parameters[0], search_method=search_method, estimate_discrete=estimate_discrete)
            results['fits'][i] = (parameters, loglikelihood)
            results['power_law_comparison'][i] = (R, p)

            R, p = distribution_compare(data, 'truncated_power_law', tpl_parameters, i, parameters, discrete, xmin, xmax)
            results['truncated_power_law_comparison'][i] = (R, p)
        return results

    #Handle edge case where we don't have enough data
    no_data = False
    if xmax and all((data > xmax) + (data < xmin)):
        #Everything is beyond the bounds of the xmax and xmin
        no_data = True
    if all(data < xmin):
        no_data = True
    if len(data) < 2:
        no_data = True
    if no_data:
        from numpy import array
        from sys import float_info
        parameters = array([0, 0, 0])
        if search_method == 'Likelihood':
            loglikelihood = -10 ** float_info.max_10_exp
        if search_method == 'KS':
            loglikelihood = 1
        if comparison_alpha is None:
            return parameters, loglikelihood
        R = 10 ** float_info.max_10_exp
        p = 1
        return parameters, loglikelihood, R, p

    n = float(len(data))

    #Initial search parameters, estimated from the data
#    print("Calculating initial parameters for search")
    if distribution == 'power_law' and not alpha:
        initial_parameters = [1 + n / sum(log(data / (xmin)))]
    elif distribution == 'exponential':
        from numpy import mean
        initial_parameters = [1 / mean(data)]
    elif distribution == 'stretched_exponential':
        from numpy import mean
        initial_parameters = [1 / mean(data), 1]
    elif distribution == 'truncated_power_law':
        from numpy import mean
        initial_parameters = [1 + n / sum(log(data / xmin)), 1 / mean(data)]
    elif distribution == 'lognormal':
        from numpy import mean, std
        logdata = log(data)
        initial_parameters = [mean(logdata), std(logdata)]
    elif distribution == 'negative_binomial':
        initial_parameters = [1, .5]
    elif distribution == 'gamma':
        from numpy import mean
        initial_parameters = [n / sum(log(data / xmin)), mean(data)]

    if search_method == 'Likelihood':
#        print("Searching using maximum likelihood method")
        #If the distribution is a continuous power law without an xmax, and we're using the maximum likelihood method, we can compute the parameters and likelihood directly
        if distribution == 'power_law' and not discrete and not xmax and not alpha:
            from numpy import array, nan
            alpha = 1 + n /\
                sum(log(data / xmin))
            loglikelihood = n * log(alpha - 1.0) - n * log(xmin) - alpha * sum(log(data / xmin))
            if loglikelihood == nan:
                loglikelihood = 0
            parameters = array([alpha])
            return parameters, loglikelihood
        elif distribution == 'power_law' and discrete and not xmax and not alpha and estimate_discrete:
            from numpy import array, nan
            alpha = 1 + n /\
                sum(log(data / (xmin - .5)))
            loglikelihood = n * log(alpha - 1.0) - n * log(xmin) - alpha * sum(log(data / xmin))
            if loglikelihood == nan:
                loglikelihood = 0
            parameters = array([alpha])
            return parameters, loglikelihood

        #Otherwise, we set up a likelihood function
        likelihood_function = likelihood_function_generator(distribution, discrete=discrete, xmin=xmin, xmax=xmax)

        #Search for the best fit parameters for the target distribution, on this data
        from scipy.optimize import fmin
        parameters, negative_loglikelihood, iter, funcalls, warnflag, = \
            fmin(
                lambda p: -sum(log(likelihood_function(p, data))),
                initial_parameters, full_output=1, disp=False)
        loglikelihood = -negative_loglikelihood

        if comparison_alpha:
            R, p = distribution_compare(data, 'power_law', [comparison_alpha], distribution, parameters, discrete, xmin, xmax)
            return parameters, loglikelihood, R, p
        else:
            return parameters, loglikelihood

    elif search_method == 'KS':
        print("Not yet supported. Sorry.")
        return
#        #Search for the best fit parameters for the target distribution, on this data
#        from scipy.optimize import fmin
#        parameters, KS, iter, funcalls, warnflag, = \
#                fmin(\
#                lambda p: -sum(log(likelihood_function(p, data))),\
#                initial_parameters, full_output=1, disp=False)
#        loglikelihood =-negative_loglikelihood
#
#        if comparison_alpha:
#            R, p = distribution_compare(data, 'power_law',[comparison_alpha], distribution, parameters, discrete, xmin, xmax)
#            return parameters, loglikelihood, R, p
#        else:
#            return parameters, loglikelihood


def distribution_compare(data, distribution1, parameters1,
        distribution2, parameters2,
        discrete, xmin, xmax, nested=None, **kwargs):
    no_data = False
    if xmax and all((data > xmax) + (data < xmin)):
        #Everything is beyond the bounds of the xmax and xmin
        no_data = True
    if all(data < xmin):
        no_data = True

    if no_data:
        R = 0
        p = 1
        return R, p

    likelihood_function1 = likelihood_function_generator(distribution1, discrete, xmin, xmax)
    likelihood_function2 = likelihood_function_generator(distribution2, discrete, xmin, xmax)

    likelihoods1 = likelihood_function1(parameters1, data)
    likelihoods2 = likelihood_function2(parameters2, data)

    if (distribution1 in distribution2) or (distribution2 in distribution1)\
        and nested==None:
        print "Assuming nested distributions"
        nested = True

    from numpy import log
    R, p = loglikelihood_ratio(log(likelihoods1), log(likelihoods2),
            nested=nested, **kwargs)

    return R, p


def likelihood_function_generator(distribution_name, discrete=False, xmin=1, xmax=None):

    if distribution_name == 'power_law':
        likelihood_function = lambda parameters, data:\
            power_law_likelihoods(
                data, parameters[0], xmin, xmax, discrete)

    elif distribution_name == 'exponential':
        likelihood_function = lambda parameters, data:\
            exponential_likelihoods(
                data, parameters[0], xmin, xmax, discrete)

    elif distribution_name == 'stretched_exponential':
        likelihood_function = lambda parameters, data:\
            stretched_exponential_likelihoods(
                data, parameters[0], parameters[1], xmin, xmax, discrete)

    elif distribution_name == 'truncated_power_law':
        likelihood_function = lambda parameters, data:\
            truncated_power_law_likelihoods(
                data, parameters[0], parameters[1], xmin, xmax, discrete)

    elif distribution_name == 'lognormal':
        likelihood_function = lambda parameters, data:\
            lognormal_likelihoods(
                data, parameters[0], parameters[1], xmin, xmax, discrete)

    elif distribution_name == 'negative_binomial':
        likelihood_function = lambda parameters, data:\
            negative_binomial_likelihoods(
                data, parameters[0], parameters[1], xmin, xmax)

    elif distribution_name == 'gamma':
        likelihood_function = lambda parameters, data:\
            gamma_likelihoods(
                data, parameters[0], parameters[1], xmin, xmax)

    return likelihood_function

def find_xmin(data, discrete=False, xmax=None, search_method='Likelihood', return_all=False, estimate_discrete=True, xmin_range=None):
    from numpy import sort, unique, asarray, argmin, vstack, arange, sqrt
    if 0 in data:
        print("Value 0 in data. Throwing out 0 values")
        data = data[data != 0]
    if xmax:
        data = data[data <= xmax]
#Much of the rest of this function was inspired by Adam Ginsburg's plfit code, specifically around lines 131-143 of this version: http://code.google.com/p/agpy/source/browse/trunk/plfit/plfit.py?spec=svn359&r=357
    if not all(data[i] <= data[i + 1] for i in xrange(len(data) - 1)):
        data = sort(data)
    if xmin_range == 'find' or xmin_range is None:
        possible_xmins = data
    else:
        possible_xmins = data[data <= max(xmin_range)]
        possible_xmins = possible_xmins[possible_xmins >= min(xmin_range)]
    xmins, xmin_indices = unique(possible_xmins, return_index=True)
    xmins = xmins[:-1]
    if len(xmins) < 2:
        from sys import float_info
        xmin = 1
        D = 1
        alpha = 0
        loglikelihood = -10 ** float_info.max_10_exp
        n_tail = 1
        noise_flag = True
        Ds = 1
        alphas = 0
        sigmas = 1

        if not return_all:
            return xmin, D, alpha, loglikelihood, n_tail, noise_flag
        else:
            return xmin, D, alpha, loglikelihood, n_tail, noise_flag, xmins, Ds, alphas, sigmas

    xmin_indices = xmin_indices[:-1]  # Don't look at last xmin, as that's also the xmax, and we want to at least have TWO points to fit!

    if search_method == 'Likelihood':
        alpha_MLE_function = lambda xmin: distribution_fit(data, 'power_law', xmin=xmin, xmax=xmax, discrete=discrete, search_method='Likelihood', estimate_discrete=estimate_discrete)
        fits = asarray(map(alpha_MLE_function, xmins))
    elif search_method == 'KS':
        alpha_KS_function = lambda xmin: distribution_fit(data, 'power_law', xmin=xmin, xmax=xmax, discrete=discrete, search_method='KS', estimate_discrete=estimate_discrete)[0]
        fits = asarray(map(alpha_KS_function, xmins))

    params = fits[:, 0]
    alphas = vstack(params)[:, 0]
    loglikelihoods = fits[:, 1]

    ks_function = lambda index: power_law_ks_distance(data, alphas[index], xmins[index], xmax=xmax, discrete=discrete)
    Ds = asarray(map(ks_function, arange(len(xmins))))

    sigmas = (alphas - 1) / sqrt(len(data) - xmin_indices + 1)
    good_values = sigmas < .1
    #Find the last good value (The first False, where sigma > .1):
    xmin_max = argmin(good_values)
    if good_values.all():  # If there are no fits beyond the noise threshold
        min_D_index = argmin(Ds)
        noise_flag = False
    elif xmin_max > 0:
        min_D_index = argmin(Ds[:xmin_max])
        noise_flag = False
    else:
        min_D_index = argmin(Ds)
        noise_flag = True

    xmin = xmins[min_D_index]
    D = Ds[min_D_index]
    alpha = alphas[min_D_index]
    loglikelihood = loglikelihoods[min_D_index]
    n_tail = sum(data >= xmin)

    if not return_all:
        return xmin, D, alpha, loglikelihood, n_tail, noise_flag
    else:
        return xmin, D, alpha, loglikelihood, n_tail, noise_flag, xmins, Ds, alphas, sigmas


def power_law_ks_distance(data, alpha, xmin, xmax=None, discrete=False, kuiper=False):
    """Helps if all data is sorted beforehand!"""
    from numpy import arange, sort, mean
    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]
    n = float(len(data))
    if n < 2:
        if kuiper:
            return 1, 1, 2
        return 1

    if not all(data[i] <= data[i + 1] for i in arange(n - 1)):
        data = sort(data)

    if not discrete:
        Actual_CDF = arange(n) / n
        Theoretical_CDF = 1 - (data / xmin) ** (-alpha + 1)

    if discrete:
        from scipy.special import zeta
        if xmax:
            bins, Actual_CDF = cumulative_distribution_function(data,xmin=xmin,xmax=xmax)
            Theoretical_CDF = 1 - ((zeta(alpha, bins) - zeta(alpha, xmax+1)) /\
                    (zeta(alpha, xmin)-zeta(alpha,xmax+1)))
        if not xmax:
            bins, Actual_CDF = cumulative_distribution_function(data,xmin=xmin)
            Theoretical_CDF = 1 - (zeta(alpha, bins) /\
                    zeta(alpha, xmin))

    D_plus = max(Theoretical_CDF - Actual_CDF)
    D_minus = max(Actual_CDF - Theoretical_CDF)
    Kappa = 1 + mean(Theoretical_CDF - Actual_CDF)

    if kuiper:
        return D_plus, D_minus, Kappa

    D = max(D_plus, D_minus)

    return D


def power_law_likelihoods(data, alpha, xmin, xmax=False, discrete=False):
    if alpha < 0:
        from numpy import tile
        from sys import float_info
        return tile(10 ** float_info.min_10_exp, len(data))

    xmin = float(xmin)
    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    if not discrete:
        likelihoods = (data ** -alpha) *\
                      ((alpha - 1) * xmin ** (alpha - 1))
    if discrete:
        if alpha < 1:
            from numpy import tile
            from sys import float_info
            return tile(10 ** float_info.min_10_exp, len(data))
        if not xmax:
            from scipy.special import zeta
            likelihoods = (data ** -alpha) /\
                zeta(alpha, xmin)
        if xmax:
            from scipy.special import zeta
            likelihoods = (data ** -alpha) /\
                          (zeta(alpha, xmin) - zeta(alpha, xmax + 1))
    from sys import float_info
    likelihoods[likelihoods == 0] = 10 ** float_info.min_10_exp
    return likelihoods


def negative_binomial_likelihoods(data, r, p, xmin=0, xmax=False):

    #Better to make this correction earlier on in distribution_fit, so as to not recheck for discreteness and reround every time fmin is used.
    #if not is_discrete(data):
    #    print("Rounding to nearest integer values for negative binomial fit.")
    #    from numpy import around
    #    data = around(data)

    xmin = float(xmin)
    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import asarray
    from scipy.misc import comb
    pmf = lambda k: comb(k + r - 1, k) * (1 - p) ** r * p ** k
    likelihoods = asarray(map(pmf, data)).flatten()

    if xmin != 0 or xmax:
        xmax = max(data)
        from numpy import arange
        normalization_constant = sum(map(pmf, arange(xmin, xmax + 1)))
        likelihoods = likelihoods / normalization_constant

    from sys import float_info
    likelihoods[likelihoods == 0] = 10 ** float_info.min_10_exp
    return likelihoods


def exponential_likelihoods(data, Lambda, xmin, xmax=False, discrete=False):
    if Lambda < 0:
        from numpy import tile
        from sys import float_info
        return tile(10 ** float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import exp
    if not discrete:
#        likelihoods = exp(-Lambda*data)*\
#                Lambda*exp(Lambda*xmin)
        likelihoods = Lambda * exp(Lambda * (xmin - data))  # Simplified so as not to throw a nan from infs being divided by each other
    if discrete:
        if not xmax:
            likelihoods = exp(-Lambda * data) *\
                             (1 - exp(-Lambda)) * exp(Lambda * xmin)
        if xmax:
            likelihoods = exp(-Lambda * data) * (1 - exp(-Lambda))\
                / (exp(-Lambda * xmin) - exp(-Lambda * (xmax + 1)))
    from sys import float_info
    likelihoods[likelihoods == 0] = 10 ** float_info.min_10_exp
    return likelihoods


def stretched_exponential_likelihoods(data, Lambda, beta, xmin, xmax=False, discrete=False):
    if Lambda < 0:
        from numpy import tile
        from sys import float_info
        return tile(10 ** float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import exp
    if not discrete:
#        likelihoods = (data**(beta-1) * exp(-Lambda*(data**beta)))*\
#            (beta*Lambda*exp(Lambda*(xmin**beta)))
        likelihoods = data ** (beta - 1) * beta * Lambda * exp(Lambda * (xmin ** beta - data ** beta))  # Simplified so as not to throw a nan from infs being divided by each other
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange
            X = arange(xmin, xmax + 1)
            PDF = X ** (beta - 1) * beta * Lambda * exp(Lambda * (xmin ** beta - X ** beta))  # Simplified so as not to throw a nan from infs being divided by each other
            PDF = PDF / sum(PDF)
            likelihoods = PDF[(data - xmin).astype(int)]
    from sys import float_info
    likelihoods[likelihoods == 0] = 10 ** float_info.min_10_exp
    return likelihoods


def gamma_likelihoods(data, k, theta, xmin, xmax=False, discrete=False):
    if k <= 0 or theta <= 0:
        from numpy import tile
        from sys import float_info
        return tile(10 ** float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import exp
    from mpmath import gammainc
#    from scipy.special import gamma, gammainc #Not NEARLY numerically accurate enough for the job
    if not discrete:
        likelihoods = (data ** (k - 1)) / (exp(data / theta) * (theta ** k) * float(gammainc(k)))
        #Calculate how much probability mass is beyond xmin, and normalize by it
        normalization_constant = 1 - float(gammainc(k, 0, xmin / theta, regularized=True))  # Mpmath's regularized option divides by gamma(k)
        likelihoods = likelihoods / normalization_constant
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange
            X = arange(xmin, xmax + 1)
            PDF = (X ** (k - 1)) / (exp(X / theta) * (theta ** k) * float(gammainc(k)))
            PDF = PDF / sum(PDF)
            likelihoods = PDF[(data - xmin).astype(int)]
    from sys import float_info
    likelihoods[likelihoods == 0] = 10 ** float_info.min_10_exp
    return likelihoods


def truncated_power_law_likelihoods(data, alpha, Lambda, xmin, xmax=False, discrete=False):
    if alpha < 0 or Lambda < 0:
        from numpy import tile
        from sys import float_info
        return tile(10 ** float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    from numpy import exp
    if not discrete:
        from mpmath import gammainc
#        from scipy.special import gamma, gammaincc #Not NEARLY accurate enough to do the job
#        likelihoods = (data**-alpha)*exp(-Lambda*data)*\
#                (Lambda**(1-alpha))/\
#                float(gammaincc(1-alpha,Lambda*xmin))
        #Simplified so as not to throw a nan from infs being divided by each other
        likelihoods = (Lambda ** (1 - alpha)) /\
                      ((data ** alpha) * exp(Lambda * data) * gammainc(1 - alpha, Lambda * xmin)).astype(float)
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange
            X = arange(xmin, xmax + 1)
            PDF = (X ** -alpha) * exp(-Lambda * X)
            PDF = PDF / sum(PDF)
            likelihoods = PDF[(data - xmin).astype(int)]
    from sys import float_info
    likelihoods[likelihoods == 0] = 10 ** float_info.min_10_exp
    return likelihoods


def lognormal_likelihoods(data, mu, sigma, xmin, xmax=False, discrete=False):
    from numpy import log
    if sigma <= 0 or mu < log(xmin):
        #The standard deviation can't be negative, and the mean of the logarithm of the distribution can't be smaller than the log of the smallest member of the distribution!
        from numpy import tile
        from sys import float_info
        return tile(10 ** float_info.min_10_exp, len(data))

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]

    if not discrete:
        from numpy import sqrt, exp
#        from mpmath import erfc
        from scipy.special import erfc
        from scipy.constants import pi
        likelihoods = (1.0 / data) * exp(-((log(data) - mu) ** 2) / (2 * sigma ** 2)) *\
            sqrt(2 / (pi * sigma ** 2)) / erfc((log(xmin) - mu) / (sqrt(2) * sigma))
#        likelihoods = likelihoods.astype(float)
    if discrete:
        if not xmax:
            xmax = max(data)
        if xmax:
            from numpy import arange, exp
#            from mpmath import exp
            X = arange(xmin, xmax + 1)
#            PDF_function = lambda x: (1.0/x)*exp(-( (log(x) - mu)**2 ) / 2*sigma**2)
#            PDF = asarray(map(PDF_function,X))
            PDF = (1.0 / X) * exp(-((log(X) - mu) ** 2) / (2 * (sigma ** 2)))
            PDF = (PDF / sum(PDF)).astype(float)
            likelihoods = PDF[(data - xmin).astype(int)]
    from sys import float_info
    likelihoods[likelihoods == 0] = 10 ** float_info.min_10_exp
    return likelihoods
