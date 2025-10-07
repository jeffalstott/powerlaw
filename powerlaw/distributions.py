import numpy as np
from numpy import nan

import warnings

import sys
import types
import scipy.optimize

from .statistics import *
from .plotting import *
from .utils import *

class Distribution(object):
    """
    An abstract class for theoretical probability distributions. Can be created
    with particular parameter values, or fitted to a dataset. Fitting is
    by maximum likelihood estimation by default.

    Parameters
    ----------
    xmin : int or float, optional
        The data value beyond which distributions should be fitted. If
        `None` an optimal one will be calculated by performing many fits
        and choosing the value that leads to the best fit.

    xmax : int or float, optional
        The maximum value of the fitted distributions.

    discrete : boolean, optional
        Whether the distribution is discrete (integers).

        Various approximations can be employed when there isn't an exact
        (or even approximate) expression for the PDF or CDF in the 
        discrete case. See `discrete_approximation` for more information.

    data : list or array, optional
        The data to which to fit the distribution. If provided, the fit will
        be created at initialization.

    fit_method : {"likelihood", "ks"}, optional
        Method for fitting the distribution. "likelihood" is maximum Likelihood
        estimation. "ks" is minimial distance estimation using the
        Kolmogorov-Smirnov test.

    parameters : tuple or list, optional
        The parameters of the distribution. If data is given, these will
        be used as the initial parameters for fitting; otherwise, they
        will be taken as the final parameters for the distribution.

    parameter_ranges : dict, optional
        Dictionary of valid parameter ranges for fitting. Formatted as a
        dictionary of parameter names (eg. 'alpha') and tuples/lists/etc.
        of their lower and upper limits (eg. (1.5, 2.5), (None, .1)).

        The use of `None` is preferred over `np.inf` to indicate an
        unbounded limit.

    discrete_approximation : "round", "xmax" or int, optional
        Approximation method to apply for a discrete distribution in the
        case that there is no analytical expression available.

        "round" sums the probability mass from `x-0.5` to `x+0.5` for each
        data point.

        The other option is to numerically normalize the probability
        distribution by summing over each x from 1 to N. If `'xmax'`, then
        N is set to be `xmax`; otherwise, the value of N should be passed
        in this kwarg.

    parent_Fit : Fit object, optional
        A Fit object from which to use data, if it exists.

    avoid_pdf_overflow : bool, optional
        Whether to represent the base PDF and it's normalization
        constant as logarithms until they are combined into a proper
        function.

        This helps to avoid overflow errors if your normalization constant
        is very large (> 1e1000) and your typical pdf values are very small
        (< 1e-1000). In principle, when you combine these values you should
        get a reasonable number, but storing the separate values can be
        difficult. The alternative to explicitly including an option like
        this is to use the `mpmath` package, but I find that this slows
        down calculations quite a bit.

        This case arises when you have a relatively large value for `xmin`,
        and are working with, for example, an exponential distribution.

    verbose : {0, 1, 2}, bool
        Whether to print debug and status information. `0` or `False` means
        print no information (including no warnings), `1` means print
        only warnings, and `2` means print warnings and status messages.
    """

    def __init__(self,
                 xmin=None,
                 xmax=None,
                 discrete=False,
                 fit_method='likelihood',
                 data=None,
                 parameters=None,
                 parameter_ranges=None,
                 discrete_approximation='round',
                 parent_Fit=None,
                 avoid_pdf_overflow=False,
                 verbose=1,
                 **kwargs):

        self.verbose = verbose

        self.xmin = xmin
        self.xmax = xmax
        self.discrete = discrete
        self.avoid_pdf_overflow = avoid_pdf_overflow
        self.fit_method = fit_method
        self.discrete_approximation = discrete_approximation

        self.data = data

        # When defining a subclass of this one, you should define this
        # list in the constructor 
        #self.parameter_names = []
        # You also will need to define this in the subclass with the
        # upper and lower bound for each parameter.
        #self.DEFAULT_PARAMETER_RANGES = {}
        # (but we don't define them here because we don't want to overwrite
        # the values created in the subclass)

        # If we don't have a parent fit, we still have to make sure that
        # this variable gets assigned, otherwise we'll have logic issues
        # later on.
        self.parent_Fit = parent_Fit

        if self.parent_Fit and not hasattr(data, '__iter__'):
            self.data = self.parent_Fit.data

        # Setup the initial parameters and things
        self.initialize_parameters(parameters)

        # Setup the parameter ranges
        # This sets the variable `self.parameter_ranges`
        self.initialize_parameter_ranges(parameter_ranges)

        # Fit if we have data
        if hasattr(self.data, '__iter__'):

            self.n = len(self.data)
            self.fit(data)

        self.debug_parameter_names()

    
    def debug_parameter_names(self):
        """
        This is solely a debug function that sets up variables in the
        old format (from v1.5, eg. self.parameter1).

        It should be removed in a future version along with an update to
        the test cases once it is convincing that the refactoring didn't
        fundamentally change anything.
        """
        self.parameter1 = None
        self.parameter2 = None
        self.parameter3 = None
        self.parameter1_name = None
        self.parameter2_name = None
        self.parameter3_name = None

        for i in range(len(self.parameter_names)): 
            # Set the name
            setattr(self, f'parameter{i+1}_name', self.parameter_names[i])

            # Set the value
            setattr(self, f'parameter{i+1}', getattr(self, self.parameter_names[i]))


    def initialize_parameters(self, initial_parameters=None):
        """
        This function sets up the parameters for the distribution. If
        parameters are passed, they will try to be parsed, otherwise
        initial guesses for parameters specific to each distribution will
        be used.

        Note that there is also a function `set_parameters()` in this class.
        The primary difference between the two is that this function allows
        the possibility of generating initial guesses of parameters
        (using the `generate_initial_parameters()` function), whereas
        the other method requires that parameter values are actually
        passed. Technically you could just use this one, but I figured
        it was more clear to have separate functions.

        Parameters
        ----------
        initial_parameters : dict or array_like, optional
            A dictionary in which each key corresponds to a parameter
            name and the value corresponds to the initial value of that
            parameters.

            Can also be given as a list that is exactly the length of
            `self.parameter_names`, and then the values in that list will
            be assumed as the initial values for the parameters in the
            same order as the former list.

            If not provided, the values will be initialized using
            `self.generate_initial_parameters()` which will try its best
            to give a reasonable start based on the data.
        """
        # If we were given a dictionary of initial parameters, we use
        # those values as is.
        if type(initial_parameters) == dict:
            assert all([k in self.parameter_names for k in initial_parameters.keys()]), f'Invalid initial parameters given: {initial_parameters}'
            initial_parameters_dict = initial_parameters

        elif hasattr(initial_parameters, '__iter__') and len(initial_parameters) == len(self.parameter_names):
            # If we are given a list of initial parameters, we assume
            # the order of them is the same as the order of self.parameter_names.
            initial_parameters_dict = dict(zip(self.parameter_names, initial_parameters))

        elif initial_parameters is None:
            # If we aren't given any initial parameters, try to generate
            # them from the data.

            data = self.data

            # Make sure that we trim our data to the defined range for
            # this distribution.
            data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)

            # If we still don't have data, we should raise an error since
            # a distribution with a prescribed exponent should never reach here.
            if not hasattr(data, '__iter__'):
                raise Exception('Trying to generate parameters without data!')

            initial_parameters_dict = self.generate_initial_parameters(data)

        else:
            # Otherwise, something must be wrong with the initial parameters
            # provided, so we raise an error.
            raise Exception(f'Invalid value provided for initial parameters: {initial_parameters}.')

        # Actually set the values
        for p in self.parameter_names:
            setattr(self, p, initial_parameters_dict[p])
        
        # We also replace initial_parameters with the proper dictionary,
        # since that is probably more useful
        self.initial_parameters = initial_parameters_dict


    def set_parameters(self, params):
        """
        This function sets the parameters for the distribution.

        Intended to be called during fitting, as opposed to `initialize_parameters()`
        which is designed to be called during construction. See documentation
        for this other function for more information.

        You could also manually set the parameters with something like:

            dist.param1 = ...
            dist.param2 = ...

        This function is totally equivalent to that, but just adds some
        parsing so you can pass the parameters in a list or dict form.

        Parameters
        ----------
        params : dict or array-like
            A dictionary in which each key corresponds to a parameter
            name and the value corresponds to the initial value of that
            parameters.

            Can also be given as a list that is exactly the length of
            `self.parameter_names`, and then the values in that list will
            be assumed as the initial values for the parameters in the
            same order as the former list.
        """
        # If we were given a dictionary of initial parameters, we use
        # those values as is.
        if type(params) == dict:
            assert all([k in self.parameter_names for k in params.keys()]), f'Invalid initial parameters given: {params}'
            params_dict = params

        elif hasattr(params, '__iter__') and len(params) == len(self.parameter_names):
            # If we are given a list of initial parameters, we assume
            # the order of them is the same as the order of self.parameter_names.
            params_dict = dict(zip(self.parameter_names, params))
        
        else:
            # Otherwise, something must be wrong with the parameters
            # provided, so we raise an error.
            raise Exception(f'Invalid value provided for setting parameters: {params}.')

        for p in self.parameter_names:
            setattr(self, p, params_dict[p])


    @property
    def parameters(self):
        """
        A dictionary of the parameters of the distribution and their
        values.

        Returns
        ----------
        params : dict
            A dictionary in which each key corresponds to a parameter
            name and the value corresponds to the initial value of that
            parameters.
        """
        return dict(zip(self.parameter_names, [getattr(self, p) for p in self.parameter_names]))


    def initialize_parameter_ranges(self, ranges=None):
        """
        This function sets up the ranges for parameters for the distribution.

        If not provided, the default range for each parameter will be
        used; these are specific to each individual distribution. For
        more information, see the variable `DEFAULT_PARAMETER_RANGES` in
        children of this class.

        Parameters
        ----------
        ranges : dict or array-like, optional
            A dictionary in which each key corresponds to a parameter
            name and the value corresponds to the lower and upper bound
            for that parameter (in the format of a length 2 tuple/list/array).

            Can also be given as a list that is exactly the length of
            `self.parameter_names`, and then the values in that list will
            be assumed as the lower and upper bound for the parameter
            with that same index in `self.parameter_names`.

            If not provided, the values will be initialized using
            `self.DEFAULT_PARAMETER_RANGES`.
        """
        # We start with the default range, since it's possible that the
        # user has passed a range for only one or two parameters instead
        # of all of them. In that case, the parameters that haven't been
        # specified should follow their default range.
        ranges_dict = self.DEFAULT_PARAMETER_RANGES

        # If we were given a dictionary of ranges, we use
        # those values as is.
        if type(ranges) == dict:
            assert all([k in self.parameter_names for k in ranges.keys()]), f'Invalid parameter ranges given: {ranges}'
            ranges_dict.update(ranges)

        # Have to make sure that ranges is a 2d list, which involves checking
        # that it has __iter__, it has at least one entry, and it's entry
        # has __iter__
        elif hasattr(ranges, '__iter__') and len(ranges) > 0 and hasattr(ranges[0], '__iter__') and len(ranges) == len(self.parameter_names):
            # If we are given a list of initial parameters, we assume
            # the order of them is the same as the order of self.parameter_names.
            ranges_dict.update(dict(zip(self.parameter_names, ranges)))

        elif ranges is None:
            # Do nothing, since we already set the default ranges above.
            pass

        else:
            # Otherwise, something must be wrong with the ranges
            # provided, so we raise an error.
            raise Exception(f'Invalid value provided for parameter ranges: {ranges}')

        self.parameter_ranges = ranges_dict


    def generate_initial_parameters(self, data=None):
        """
        This function should be implemented in child classes to create an
        initial guess for each parameter based on the data passed either
        directly to this function, or to the class instance.

        This can't be implemented in this class because the specific
        parameters and their values will depend on what type of 
        distribution you are working with.

        Returns
        -------
        params : dict
            A dictionary where the keys are the parameter names and the values
            are the initial values of that parameter.
        """
        # Of course remove this line when you reimplement this function.
        raise NotImplementedError('generate_initial_parameters() not implemented')

        params = {}

        # Initialize your parameters here
        # ...

        # For example, for a power law, assuming f is a function that
        # guesses the exponent based on the data.
        # params["alpha"] = f(data)

        return params


    def fit(self, data=None):
        """
        Fits the parameters of the distribution to the data. Uses options set
        at initialization.

        Fitting is performed by minimizing either the loglikelihood or
        KS distance (depending on the value of `Distribution.fit_method`).

        Bounds defined in `self.parameter_ranges` are used explicitly during
        the minimization process.

        Parameters
        ----------
        data : array_like, optional
            The data to fit the distribution to, if different from the
            data used to initialize the class.

        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)

        # Define our cost function based on the fitting method we've chosen.
        if self.fit_method.lower() == 'likelihood':

            def fit_function(params):
                # Set the parameters
                self.set_parameters(params)
                # Compute log likelihood
                cost = -np.sum(self.loglikelihoods(data))
                return cost

        elif self.fit_method.lower() == 'ks':

            def fit_function(params):
                # Set the parameters
                self.set_parameters(params)
                # Compute Kolmogorov-Smirnov 
                self.compute_distance_metrics()
                cost = self.D
                return cost

        # Format the bounds as required for scipy's minimize
        bounds = [b for b in self.parameter_ranges.values()]

        # TODO: Add constraints here
        # In choosing the minimization method, unfortunately there isn't
        # one choice that works for all cases. Powell and Nelder-Mead 
        # together cover most options though is what I've found.
        # In particular, Powell can traverse the inflection point at
        # alpha = 1 only going down (ie works for any alpha <= 1) whereas
        # Nelder-Mead can only traverse it going up (ie. works for any
        # alpha > 1).

        # The only options I've found that have success both ways is
        # COBYLA and COBYQA. COBYQA struggles around alpha = 1, but
        # COBYLA seems relatively fine.

        # Maybe it sounds dumb, but an alternative is to switch between
        # two algorithms a few times, since, for example, both Nelder-Mead and Powell
        # will work in each domain (so long as they don't need to traverse
        # alpha = 1).
        # This combination works well for values close to 1 from below
        # (eg. 0.98) but not for values from above (eg. 1.02). Maybe
        # there is some way to fix this; I think the issue stems from
        # inaccuracy in the pdf normalization. For now, this can be
        # sort of addressed by using `discrete=True` since this just
        # calculates the normalization numerically.

        # After testing, I think going with COBYLA is the fastest and most
        # accurate option, but this should definitely be reviewed later on.
        switches = 1
        # The order doesn't really matter here.
        #methods = ['Powell', 'Nelder-Mead']
        methods = ["COBYLA"]

        # Take the initial parameters
        parameters = list(self.parameters.values())

        for i in range(switches):
            for m in methods:
                result = scipy.optimize.minimize(fit_function,
                                                 x0=parameters,
                                                 bounds=bounds,
                                                 method=m,
                                                 tol=1e-4)
                parameters = result.x

        # In case you'd like to compare, this uses the exact same optimization
        # as in powerlaw v1.5 and before. The results end up pretty much
        # the same as with the new method above. This method is slightly
        # faster than the one above when using two methods above and 2
        # switches. COBYLA is comparable in speed.
        #self.initialize_parameters()
        #initial_parameters = list(self.parameters.values())

        #parameters, negative_loglikelihood, iter, funcalls, warnflag, = \
        #        scipy.optimize.fmin(
        #                        lambda params: fit_function(params),
        #                        initial_parameters,
        #                        full_output=1,
        #                        disp=False)


        # Save the optimized parameters
        self.set_parameters(parameters)

        # Flag as noisy (not fit) if the parameters aren't in range
        self.noise_flag = (not self.in_range())# or (not result.success)
  
        if self.noise_flag and self.verbose:
            warnings.warn("No valid fits found.")

        # Recompute goodness of fit metrics
        self.loglikelihood = np.sum(self.loglikelihoods(data))
        self.compute_distance_metrics(data)

        # Give a warning if the fit parameters are very close to the
        # boundaries, indicating that the parameter ranges are probably
        # wrong.

        # The small value to check if we are close to the boundary.
        # Doesn't actually need to be that small, just enough to see that
        # the value is close to the edge.
        eps = 1e-2
        nearBoundary = False
        # TODO Should there be different behavior is a limit is 0? So that
        # we don't trigger this warning just from having a (possibly totally
        # reasonable) small value? Maybe make eps scale with the magnitude
        # of the parameter value?

        for p in self.parameter_names:

            if self.parameter_ranges[p][0] is not None:
                nearBoundary += getattr(self, p) - eps < self.parameter_ranges[p][0]

            if self.parameter_ranges[p][1] is not None:
                nearBoundary += getattr(self, p) + eps > self.parameter_ranges[p][1]

        if nearBoundary:
            self.noise_flag = True

            if self.verbose:
                warnings.warn('Fitted parameters are very close to the edge of parameter ranges; consider changing these ranges.')


    def KS(self, data=None):
        """
        Return the Kolmogorov-Smirnov distance D.

        Included for backwards compatability; this method has been
        renamed `compute_distance_metrics()` because it compute several
        distance metrics (including KS).
        """
        compute_distance_metrics(data)
        return self.D

    
    def compute_distance_metrics(self, data=None):
        r"""
        Compute various distance metrics between the fit distribution and
        the actual distribution of data.

        The following distance metrics will be computed, with $C_d$ and $C_t$
        as the cumulative distribution function for the data and for the
        theoretical fit, respectively.

        Kolmogorov-Smirnov distance:
        $$ D = max( max(C_d - C_t), -min(C_d - C_t) ) $$
         
        Kuiper distance:
        $$ V = max(C_d - C_t) - min(C_d - C_t) $$

        Anderson-Darling distance:
        $$ A^2 = \sum( (C_d - C_t)^2 / (C_t (1 - C_t))) $$

        Kappa:
        $$ K = 1 + mean(C_d - C_t) $$

        The names of these distance metrics will be stored as `D`, `V`, 
        `Asquare`, and `Kappa` respectively.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)

        # If we don't have enough data, return a bunch of nan values
        if len(data) < 2:
            warnings.warn("Not enough data to compute KS, returning nan")

            self.D = nan
            self.D_plus = nan
            self.D_minus = nan
            self.Kappa = nan
            self.V = nan
            self.Asquare = nan
            return self.D

        # In order to compute KS and other distance metrics, we need the
        # cumulative distribution function for the real data, as well as
        # for the fit distribution.

        # If we have a parent fit, the one for the real data should already
        # have been calculated.
        if self.parent_Fit:
            bins = self.parent_Fit.fitting_cdf_bins
            Actual_CDF = self.parent_Fit.fitting_cdf
            # But since xmin may be specific to this distribution, not the
            # parent fit, we need to do the filtering here.
            ind = bins >= self.xmin
            bins = bins[ind]
            Actual_CDF = Actual_CDF[ind]

            # And we remove all the probability that was contained in
            # bins before xmin
            dropped_probability = Actual_CDF[0]
            Actual_CDF -= dropped_probability
            Actual_CDF /= 1 - dropped_probability

        else:
            # If we don't have the cdf already computed, we compute it now
            bins, Actual_CDF = cdf(data)

        # Now compute the theoretical cdf of the fit distribution
        Theoretical_CDF = self.cdf(bins)

        CDF_diff = Theoretical_CDF - Actual_CDF

        # Compute the various metrics of distance between the two
        # distributions based on the difference in CDFs
        self.D_plus = np.max(CDF_diff)
        self.D_minus = -np.min(CDF_diff)

        # Not sure if this metric has a proper name, it's not mentioned
        # anywhere in the documentation.
        self.Kappa = 1 + np.mean(CDF_diff)

        # Kolmogorov-Smirnov distance, D
        # Insensitive to differences at the tails of the distributions.
        self.D = max(self.D_plus, self.D_minus)

        # Kuiper distance, V
        # Gives additional weight to the tails, but mostly performs the
        # same as the KS distance
        self.V = self.D_plus + self.D_minus

        # Anderson-Darling distance, Asquare
        # Very conservative distance metric, so only works well with
        # lots of points.
        self.Asquare = np.sum((
                            (CDF_diff**2) /
                            (Theoretical_CDF * (1 - Theoretical_CDF) + 1e-12)
                            )[1:]
                             )

        # We don't return anything, just compute the values


    def ccdf(self,data=None, survival=True):
        """
        The complementary cumulative distribution function (CCDF) of the
        theoretical distribution. Calculated for the values given in data
        within xmin and xmax, if present.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        survival : bool, optional
            Whether to calculate a CDF (False) or CCDF (True).
            True by default.

        Returns
        -------
        X : array
            The sorted, unique values in the data.
        probabilities : array
            The portion of the data that is less than or equal to X.
        """
        return self.cdf(data=data, survival=survival)


    def cdf(self,data=None, survival=False):
        """
        The cumulative distribution function (CDF) of the theoretical
        distribution. Calculated for the values given in data within xmin and
        xmax, if present.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.

        survival : bool, optional
            Whether to calculate a CDF (False) or CCDF (True). False by
            default.

            See wrapper method `Distribution.ccdf()`.

        Returns
        -------
        X : array
            The sorted, unique values in the data.

        probabilities : array
            The portion of the data that is less than or equal to X.
        """
        # TODO: Clean this up
        if data is None and self.parent_Fit:
            data = self.parent_Fit.data

        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        n = len(data)
        from sys import float_info
        if not self.in_range():
            return np.tile(10**float_info.min_10_exp, n)

        if self._cdf_xmin == 1:
            # If cdf_xmin is 1, it means we don't have the numerical accuracy to
            # calculate this tail. So we make everything 1, indicating
            # we're at the end of the tail. Such an xmin should be thrown
            # out by the KS test.
            CDF = np.ones(n)
            return CDF

        CDF = self._cdf_base_function(data) - self._cdf_xmin

        norm = 1 - self._cdf_xmin
        if self.xmax:
            norm = norm - (1 - self._cdf_base_function(self.xmax))

        CDF = CDF/norm

        if survival:
            CDF = 1 - CDF

        # If we have any nan values in the cdf, it is indicative of a
        # numerical error, so we should warn.
        # np.min will always give nan if there are any nans
        possible_numerical_error = np.isnan(np.min(CDF))

        if possible_numerical_error and self.verbose:
            warnings.warn("Likely underflow or overflow error: the optimal fit for this distribution gives values that are so extreme that we lack the numerical precision to calculate them.")

        return CDF


    @property
    def _cdf_xmin(self):
        return self._cdf_base_function(self.xmin)


    def pdf(self, data=None):
        """
        Returns the probability density function (normalized histogram) of the
        theoretical distribution for the values in data within xmin and xmax,
        if present.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.

        Returns
        -------
        probabilities : array
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)

        n = len(data)

        from sys import float_info
        # I guess this is an attempt to return very small values that aren't
        # zeros since that would mess with the log scale
        if not self.in_range():
            return np.tile(10**float_info.min_10_exp, n)

        # If we have a continuous distribution, we can easily apply
        # our base function and the normalization factor to get the
        # pdf evaluated at each data point.
        if not self.discrete:
            f = self._pdf_base_function(data)
            C = self._pdf_continuous_normalizer

            likelihoods = f*C

            # In some cases, the normalization and typical pdf values will
            # be very extreme on their own, but will combine to be normal
            # values. In this case, we represent the separate values as
            # the log of the real value, and then take the exp here only
            # after they've been combined.
            if self.avoid_pdf_overflow:
                likelihoods = np.exp(likelihoods)

        else:
            # For discrete cases, it's a little more tricky since we will
            # have to approximate the normalization factor.
            if self._pdf_discrete_normalizer:
                # If we have an explicit expression for the discrete
                # normalization, we should of course use that.
                f = self._pdf_base_function(data)
                C = self._pdf_discrete_normalizer

                likelihoods = f*C

                # Also have to check if we represent the pdf as itself or as
                # its logarithm.
                if self.avoid_pdf_overflow:
                    likelihoods = np.exp(likelihoods)

            elif self.discrete_approximation=='round':
                # If we want to approximate by rounding values, we essentially
                # take a discretized derivative of the cumulative distribution
                # function.
                lower_data = data - 0.5
                upper_data = data + 0.5

                # Temporarily expand xmin and xmax to be able to grab the extra bit of
                # probability mass beyond the (integer) values of xmin and xmax
                # Note this is a design decision. One could also say this extra
                # probability "off the edge" of the distribution shouldn't be included,
                # and that implementation is retained below, commented out. Note, however,
                # that such a cliff means values right at xmin and xmax have half the width to
                # grab probability from, and thus are lower probability than they would otherwise
                # be. This is particularly concerning for values at xmin, which are typically
                # the most likely and greatly influence the distribution's fit.
                self.xmin -= 0.5
                if self.xmax:
                    self.xmax += 0.5

                # Clean data for invalid values before handing to cdf, which will purge them
                #lower_data[lower_data<self.xmin] +=.5
                #if self.xmax:
                #    upper_data[upper_data>self.xmax] -=.5
                likelihoods = self.cdf(upper_data) - self.cdf(lower_data)

                self.xmin += 0.5
                if self.xmax:
                    self.xmax -= 0.5
            else:
                # Otherwise, we just normalize numerically by summing the
                # PDF
                if self.discrete_approximation=='xmax':
                    upper_limit = self.xmax
                else:
                    upper_limit = self.discrete_approximation

                X = np.arange(self.xmin, upper_limit+1)
                PDF = self._pdf_base_function(X)

                if self.avoid_pdf_overflow:
                    PDF = np.exp(PDF)

                PDF = (PDF / np.sum(PDF)).astype(float)
                likelihoods = PDF[(data - self.xmin).astype(int)]

        # Set any zeros to just very small values
        likelihoods[likelihoods == 0] = 10**float_info.min_10_exp

        return likelihoods


    @property
    def _pdf_continuous_normalizer(self):
        """
        This is the (inverse of the) normalization constant for the pdf
        assuming a continuous distribution. So you can get the normalized
        pdf evaluated for a particular x as:

            _pdf_continuous_normalizer * _pdf_base_function(x)

        The implementation below simply uses the cumulative distribution
        function to compute this, though if an explicit expression can
        be found by integrating the pdf, this function should be overwritten
        in child classes.
        """
        # Essentially equivalent to numerically integrating the pdf over
        # the whole domain.
        C = 1 - self._cdf_xmin

        # If the upper limit of the normalization is xmax instead of
        # infinity, account for that.
        if self.xmax:
            C -= 1 - self._cdf_base_function(self.xmax + 1)

        # Take the inverse so we can just multiply
        return 1. / C


    @property
    def _pdf_discrete_normalizer(self):
        """
        This is the (inverse of the) normalization constant for the pdf
        assuming a discrete distribution.

        It is not currently implemented since the value will depend on your
        specific distribution, so it should be implemented in child classes.
        """
        return False


    def in_range(self):
        """
        Whether the current parameters (`self.parameters`) of the
        distribution are within the range of valid parameters (defined by
        `self.parameter_ranges`).

        Returns
        -------
        result : bool
            True if all parameters (defined by `self.parameter_names`) are
            within the ranges specified by `self.parameter_ranges`.
        """
        # Final result of whether all of the parameters are in range
        result = True

        for p in self.parameter_names:
            if self.parameter_ranges[p][0] is not None:
                result *= getattr(self, p) > self.parameter_ranges[p][0]
            if self.parameter_ranges[p][1] is not None:
                result *= getattr(self, p) < self.parameter_ranges[p][1]

        return bool(result)


    def likelihoods(self, data=None):
        """
        The likelihoods of the observed data from the theoretical distribution.
        Another name for the probabilities or probability density function.

        Parameters
        ----------
        data : array_like, optional
            The data to compute the likelihood of.

            If not provided, data used to create the Distribution object
            will be used (if available).

        Returns
        -------
        likelihoods : array_like
            Likelihood of each observed data.

        """
        # No need to trim to range or check if data is None because pdf()
        # does that.
        return self.pdf(data)


    def loglikelihoods(self, data=None):
        """
        The logarithm of the likelihoods of the observed data from the
        theoretical distribution.

        Parameters
        ----------
        data : array_like, optional
            The data to compute the likelihood of.

            If not provided, data used to create the Distribution object
            will be used (if available).

        Returns
        -------
        likelihoods : array_like
            Logarithm (base e) of likelihood of each observed data.
        """
        # No need to trim to range or check if data is None because pdf()
        # does that, which is called by self.likelihoods().
        return np.log(self.likelihoods(data))


    def plot_ccdf(self, data=None, ax=None, **kwargs):
        """
        Plots the complementary cumulative distribution function (CDF) of the
        theoretical distribution for the values given in data within xmin and
        xmax, if present. Plots to a new figure or to axis ax if provided.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.

        ax : matplotlib axis, optional
            The axis on which to plot. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        # This function used to have `survival` as a keyword, but that is
        # redundant because there are already two separate functions for
        # cdf and ccdf.
        return self.plot_cdf(data, ax=ax, survival=True, **kwargs)


    def plot_cdf(self, data=None, ax=None, survival=False, **kwargs):
        """
        Plots the cumulative distribution function (CDF) of the
        theoretical distribution for the values given in data within xmin and
        xmax, if present. Plots to a new figure or to axis ax if provided.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.

        ax : matplotlib axis, optional
            The axis on which to plot. If None, a new figure is created.

        survival : bool, optional
            Whether to plot a CDF (False) or CCDF (True). False by default.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        bins = np.unique(trim_to_range(data, xmin=self.xmin, xmax=self.xmax))
        CDF = self.cdf(bins, survival=survival)

        if not ax:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

        ax.plot(bins, CDF, **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")

        return ax


    def plot_pdf(self, data=None, ax=None, **kwargs):
        """
        Plots the probability density function (PDF) of the
        theoretical distribution for the values given in data within xmin and
        xmax, if present. Plots to a new figure or to axis ax if provided.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        bins = np.unique(trim_to_range(data, xmin=self.xmin, xmax=self.xmax))
        PDF = self.pdf(bins)

        # Set to nan so it doesn't show up on the plot
        PDF[PDF == 0] = nan

        if not ax:
            import matplotlib.pyplot as plt
            plt.plot(bins, PDF, **kwargs)
            ax = plt.gca()

        else:
            ax.plot(bins, PDF, **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")

        return ax


    def generate_random(self,n=1, estimate_discrete=None):
        """
        Generates random numbers from the theoretical probability distribution.
        If xmax is present, it is currently ignored.

        Parameters
        ----------
        n : int or float
            The number of random numbers to generate
        estimate_discrete : boolean
            For discrete distributions, whether to use a faster approximation of
            the random number generator. If None, attempts to inherit
            the estimate_discrete behavior used for fitting from the Distribution
            object or the parent Fit object, if present. Approximations only
            exist for some distributions (namely the power law). If an
            approximation does not exist an estimate_discrete setting of True
            will not be inherited.

        Returns
        -------
        r : array
            Random numbers drawn from the distribution
        """
        # TODO: Clean this up
        from numpy.random import rand
        from numpy import array
        r = rand(n)
        if not self.discrete:
            x = self._generate_random_continuous(r)
        else:
            if (estimate_discrete and not hasattr(self, '_generate_random_discrete_estimate') ):
                raise AttributeError("This distribution does not have an "
                                     "estimation of the discrete form for generating simulated "
                                     "data. Try the exact form with estimate_discrete=False.")
            if estimate_discrete is None:
                if not hasattr(self, '_generate_random_discrete_estimate'):
                    estimate_discrete = False
                elif hasattr(self, 'estimate_discrete'):
                    estimate_discrete = self.estimate_discrete
                elif self.parent_Fit:
                    estimate_discrete = self.parent_Fit.estimate_discrete
                else:
                    estimate_discrete = False
            if estimate_discrete:
                x = self._generate_random_discrete_estimate(r)
            else:
                x = array([self._double_search_discrete(R) for R in r],
                          dtype='float')
        return x

    def _double_search_discrete(self, r):
        #Find a range from x1 to x2 that our random probability fits between
        x2 = int(self.xmin)
        while self.ccdf(data=[x2]) >= (1 - r):
            x1 = x2
            x2 = 2*x1
        #Use binary search within that range to find the exact answer, up to
        #the limit of being between two integers.
        x = bisect_map(x1, x2, self.ccdf, 1-r)
        return x


class Power_Law(Distribution):

    def __init__(self, estimate_discrete=True, pdf_ends_at_xmax=False, **kwargs):
        """
        A power law distribution with form:

            $$ p(x) ~ x^{-alpha} $$

        The exponent alpha should be positive. For a normalizable distribution
        on an infinite domain (no xmax), we should have alpha greater than 1.

        Accepts all kwargs from `Distribution` super class.

        Parameters
        ----------
        estimate_discrete : bool

        pdf_ends_at_xmax : bool

        """

        self.parameter_names = ['alpha']
        self.DEFAULT_PARAMETER_RANGES = {'alpha': [0, 3]}

        self.estimate_discrete = estimate_discrete
        self.pdf_ends_at_xmax = pdf_ends_at_xmax

        Distribution.__init__(self, **kwargs)

        # We will now have a fitted distribution, and we might want to
        # warn the user if we fit an exponent close to 1 from above.
        # For more information on this, see the discussion in
        # Distribution.fit().
        if self.alpha > 1.0 and self.alpha < 1.2 and not self.discrete and self.verbose:
            warnings.warn('Fit detected an alpha value slightly above one. Fitting algorithms in this regime are \
                          error-prone; it might help to set `discrete=True` to numerically calculate the normalization.')


    def generate_initial_parameters(self, data):
        r"""
        Generate initial guesses for the distribution parameters based
        on the data.

        For continuous distributions, we use the following value to estimate
        alpha (see Clauset et al. (2009) [1], Eq. 3.1). In the limit as N goes to infinity,
        this becomes exact, and generally is a very good estimation even
        at modest values (~1000) of N.

            $$ \alpha_0 = 1 + N / ( \sum \log (x / x_{min})) $$

        For the discrete case, there is no form that is exact in the large
        N limit, but the following value is a good approximation (see ref
        [1], Eq. 3.7).

            $$ \alpha_0 = 1 + N / ( \sum \log (x / (x_{min} - 1/2))) $$


        Parameters
        ----------
        data : array_like
            The data to use to generate the initial values of parameters.

            Should already be trimmed to the data range defined by
            `xmin` and `xmax` (if included).

        Returns
        -------
        params : dict
            A dictionary of the parameters and their values.

        References
        ----------
        [1] Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).
        Power-law distributions in empirical data. SIAM Review, 51(4),
        661–703. https://doi.org/10.1137/070710111

        """
        params = {}

        # This is generally a very good approximation of the power law
        # exponent for alpha > 1. For values of alpha < 1, it will only
        # approach 1, as expected from the 1 + ...

        n = len(data)

        # If we have a discrete distribution (ie only takes on integer
        # values) we have to shift slightly.
        if self.discrete and self.estimate_discrete and not self.xmax:
            params["alpha"] = 1 + n / np.sum(np.log(data / (self.xmin - 0.5)))

        else:
            # For continuous, we just have the usual expression.
            params["alpha"] = 1 + n / np.sum(np.log(data / (self.xmin)))

        return params


    @property
    def name(self):
        return "power_law"


    @property
    def sigma(self):
        """
        The standard error of the MLE.
        """
        # Only is calculable after self.fit is started, when the number of data points is
        # established
        return (self.alpha - 1) / np.sqrt(self.n)


    def _cdf_base_function(self, x):
        if self.discrete:
            from scipy.special import zeta
            CDF = 1 - zeta(self.alpha, x)
        else:
            #Can this be reformulated to not reference xmin? Removal of the probability
            #before xmin and after xmax is handled in Distribution.cdf(), so we don't
            #strictly need this element. It doesn't hurt, for the moment.
            CDF = 1 - (x / self.xmin)**(-self.alpha + 1)
        return CDF


    def _pdf_base_function(self, x):
        return x**(-self.alpha)


    @property
    def _pdf_continuous_normalizer(self):
        # The pdf has a different form when we consider xmax as
        # the upper limit of the distribution. When alpha < 1, we have to
        # assume the distribution ends at xmax otherwise it is not
        # normalizable.
        if self.alpha <= 1 and not self.xmax:
            if self.verbose:
                warnings.warn('Distribution with alpha <= 1 has no xmax; setting xmax to be max(data) otherwise cannot continue. Consider setting an explicit value for xmax')
            xmax = np.max(self.data)
        else:
            xmax = self.xmax

        if self.pdf_ends_at_xmax or self.alpha <= 1:
            return (1 - self.alpha)/(xmax**(1 - self.alpha) - self.xmin**(1 - self.alpha))
        else:
            return (self.alpha - 1) * self.xmin**(self.alpha - 1)


    @property
    def _pdf_discrete_normalizer(self):
        C = 1.0 - self._cdf_xmin
        if self.xmax:
            C -= 1 - self._cdf_base_function(self.xmax+1)
        C = 1.0/C
        return C


    def _generate_random_continuous(self, r):
            return self.xmin * (1 - r) ** (-1/(self.alpha - 1))


    def _generate_random_discrete_estimate(self, r):
            x = (self.xmin - 0.5) * (1 - r) ** (-1/(self.alpha - 1)) + 0.5
            from numpy import around
            return around(x)


class Exponential(Distribution):

    def __init__(self, **kwargs):
        r"""
        An exponential distribution, with pdf:

            $$ p(x) ~ exp(- \lambda x) $$

        For expressions for normalization for discrete and continuous
        distributions, see Clauset et al. (2009) [1], Table 2.1.

        References
        ----------
        [1] Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).
        Power-law distributions in empirical data. SIAM Review, 51(4),
        661–703. https://doi.org/10.1137/070710111
        """
        # Note that Lambda has a capital L; I guess this is because
        # the authors didn't want to cause confusion with Python's lambda
        # expressions.
        self.parameter_names = ['Lambda']
        self.DEFAULT_PARAMETER_RANGES = {'Lambda': [0, None]}

        # Note that we use the option avoid_pdf_overflow which means
        # that our _pdf_base_function is actually the logarithm of
        # the base function, and same for the normalizer.
        Distribution.__init__(self, avoid_pdf_overflow=False, **kwargs)


    @property
    def name(self):
        return "exponential"


    def generate_initial_parameters(self, data):
        r"""
        For an exponential distribution, we estimate the exponent factor
        as:
            $$ \lambda_0 = 1 / mean(x) $$


        Parameters
        ----------
        data : array_like
            The data to use to generate the initial values of parameters.

            Should already be trimmed to the data range defined by
            `xmin` and `xmax` (if included).

        Returns
        -------
        params : dict
            A dictionary of the parameters and their values.
        """
        params = {}

        params["Lambda"] = 1 / np.mean(data)

        return params


    def _cdf_base_function(self, x):
        r"""
        The cumulative distribution function for an exponential distribution
        is:
            $$ c(x) ~ 1 - exp(-\lambda x) $$
        """
        return 1 - np.exp(-self.Lambda*x)


    def _pdf_base_function(self, x):
        r"""
        The probability distribution function for an exponential distribution
        is:
            $$ p(x) ~ exp(-\lambda x) $$

        Note that we use the option `self.avoid_pdf_overflow=True` which
        means that this should return actually the logarithm of the pdf.
        """
        return np.exp(-self.Lambda * x)
        #return -self.Lambda * x


    @property
    def _pdf_continuous_normalizer(self):
        """
        Note that we use the option `self.avoid_pdf_overflow=True` which
        means that this should return actually the logarithm of the pdf.
        """
        # We could put an expression including xmax here but it probably
        # wouldn't make much of a difference.
        
        # There's a reasonable chance this line can lead to overflow errors
        # since it is a positive exponential with what could be a large
        # number. The end calculation would theoretically be fine since
        # the x values will always be > xmin, and therefore we will also
        # have a tiny _pdf_base_function value
        return self.Lambda * np.exp(self.Lambda * self.xmin)
        #return np.log(self.Lambda) + self.Lambda * self.xmin


    @property
    def _pdf_discrete_normalizer(self):
        """
        Note that we use the option `self.avoid_pdf_overflow=True` which
        means that this should return actually the logarithm of the pdf.
        """
        C = (1 - np.exp(-self.Lambda)) * np.exp(self.Lambda * self.xmin)
        #C = np.log(1 - np.exp(-self.Lambda)) + self.Lambda * self.xmin

        if self.xmax:
            Cxmax = (1 - np.exp(-self.Lambda)) * np.exp(self.Lambda * self.xmax)
            C = 1.0/C - 1.0/Cxmax
            C = 1.0/C

            #Cxmax = np.log(1 - np.exp(-self.Lambda)) + self.Lambda * self.xmax

        return C


    # This function used to overload the pdf() defined in the super class
    # but it doesn't seem to do anything different...?
    # If the if condition evaluates true, then it just calculated the
    # likelihood as _pdf_base_function(x) * _pdf_continuous_normalizer,
    # which is exactly what the parent pdf() does.
#    def pdf(self, data=None):
#        if data is None and self.parent_Fit:
#            data = self.parent_Fit.data
#
#        if not self.discrete and self.in_range() and not self.xmax:
#            print('special pdf2')
#            data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
#            from numpy import exp
#        likelihoods = exp(-Lambda*data)*\
#                Lambda*exp(Lambda*xmin)
#
#            # This is _pdf_base_function(x) * _pdf_continuous_normalizer(xmin)
#            likelihoods = self.Lambda*exp(self.Lambda*(self.xmin-data))
#
#            #Simplified so as not to throw a nan from infs being divided by each other
#            from sys import float_info
#            likelihoods[likelihoods==0] = 10**float_info.min_10_exp
#
#        else:
#            likelihoods = Distribution.pdf(self, data)
#
#        return likelihoods
#
#    def loglikelihoods(self, data=None):
#        if data is None and self.parent_Fit:
#            data = self.parent_Fit.data
#
#        if not self.discrete and self.in_range() and not self.xmax:
#            data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
#            from numpy import log
#        likelihoods = exp(-Lambda*data)*\
#                Lambda*exp(Lambda*xmin)
#            loglikelihoods = log(self.Lambda) + (self.Lambda*(self.xmin-data))
#            #Simplified so as not to throw a nan from infs being divided by each other
#            from sys import float_info
#            loglikelihoods[loglikelihoods==0] = log(10**float_info.min_10_exp)
#        else:
#            loglikelihoods = Distribution.loglikelihoods(self, data)
#        return loglikelihoods


    def _generate_random_continuous(self, r):
        return self.xmin - (1/self.Lambda) * np.log(1-r)


class Stretched_Exponential(Distribution):

    def __init__(self, **kwargs):
        r"""
        A stretched exponential distribution, with pdf:

            $$ p(x) ~ (x \lambda)^{\beta - 1} exp(- (\lambda x)^\beta) $$

        For expressions for normalization for discrete and continuous
        distributions, see Clauset et al. (2009) [1], Table 2.1.

        References
        ----------
        [1] Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).
        Power-law distributions in empirical data. SIAM Review, 51(4),
        661–703. https://doi.org/10.1137/070710111
        """
        # Note that Lambda has a capital L; I guess this is because
        # the authors didn't want to cause confusion with Python's lambda
        # expressions.
        # That being said, the old code had:
        # self.parameter1 = self.Lambda
        # self.parameter1_name = 'lambda'
        # Which is inconsistent; I've chosen the capital lambda so far,
        # but should look into making this more consistent later. TODO

        self.parameter_names = ['Lambda', 'beta']
        self.DEFAULT_PARAMETER_RANGES = {'Lambda': [0, None],
                                         'beta': [0, None]}

        Distribution.__init__(self, **kwargs)


    @property
    def name(self):
        return "stretched_exponential"


    def generate_initial_parameters(self, data):
        r"""
        For an exponential distribution, we estimate the exponent factor
        as:
            $$ \lambda_0 = 1 / mean(x) $$

        The stretch exponent $\beta$ just starts with a value $1$.

        Parameters
        ----------
        data : array_like
            The data to use to generate the initial values of parameters.

            Should already be trimmed to the data range defined by
            `xmin` and `xmax` (if included).

        Returns
        -------
        params : dict
            A dictionary of the parameters and their values.
        """
        params = {}

        params["Lambda"] = 1 / np.mean(data)
        params["beta"] = 1

        return params


    def _cdf_base_function(self, x):
        from numpy import exp
        CDF = 1 - np.exp(-(self.Lambda*x)**self.beta)
        return CDF


    def _pdf_base_function(self, x):
        # TODO: This is different from the base function defined in Clauset
        # et al. 2009. It has extra factors of lambda and exp(lambda^beta)
        from numpy import exp
        return (((x*self.Lambda)**(self.beta-1)) *
                exp(-((self.Lambda*x)**self.beta)))


    @property
    def _pdf_continuous_normalizer(self):
        from numpy import exp
        C = self.beta*self.Lambda*exp((self.Lambda*self.xmin)**self.beta)
        return C


    @property
    def _pdf_discrete_normalizer(self):
        return False


    # These are deprecated and do nothing different than the super
    # implementation in Distribution
#    def pdf(self, data=None):
#        if data is None and self.parent_Fit:
#            data = self.parent_Fit.data
#
#        if not self.discrete and self.in_range() and not self.xmax:
#            data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
#            from numpy import exp
#            likelihoods = ((data*self.Lambda)**(self.beta-1) *
#                           self.beta * self.Lambda *
#                           exp((self.Lambda*self.xmin)**self.beta -
#                               (self.Lambda*data)**self.beta))
#            #Simplified so as not to throw a nan from infs being divided by each other
#            from sys import float_info
#            likelihoods[likelihoods==0] = 10**float_info.min_10_exp
#        else:
#            likelihoods = Distribution.pdf(self, data)
#        return likelihoods
#
#    def loglikelihoods(self, data=None):
#        if data is None and self.parent_Fit:
#            data = self.parent_Fit.data
#
#        if not self.discrete and self.in_range() and not self.xmax:
#            data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
#            from numpy import log
#            loglikelihoods = (
#                    log((data*self.Lambda)**(self.beta-1) *
#                        self.beta * self. Lambda) +
#                    (self.Lambda*self.xmin)**self.beta -
#                        (self.Lambda*data)**self.beta)
#            #Simplified so as not to throw a nan from infs being divided by each other
#            from sys import float_info
#            from numpy import inf
#            loglikelihoods[loglikelihoods==-inf] = log(10**float_info.min_10_exp)
#        else:
#            loglikelihoods = Distribution.loglikelihoods(self, data)
#        return loglikelihoods

    def _generate_random_continuous(self, r):
        from numpy import log
#        return ( (self.xmin**self.beta) -
#            (1/self.Lambda) * log(1-r) )**(1/self.beta)
        return (1/self.Lambda)* ( (self.Lambda*self.xmin)**self.beta -
            log(1-r) )**(1/self.beta)


class Truncated_Power_Law(Distribution):

    def __init__(self, **kwargs):
        r"""
        A power law distribution truncated by an exponential with form:

            $$ p(x) ~ x^{-alpha} exp(-\lambda x)$$

        The exponent alpha should be positive. 

        Parameters
        ----------
        estimate_discrete : bool

        pdf_ends_at_xmax : bool

        """

        self.parameter_names = ['alpha', 'Lambda']
        self.DEFAULT_PARAMETER_RANGES = {'alpha': [0, 3],
                                         'Lambda': [0, None]}

        Distribution.__init__(self, **kwargs)


    @property
    def name(self):
        return "truncated_power_law"


    def generate_initial_parameters(self, data):
        r"""
        Generate initial guesses for the distribution parameters based
        on the data.

        For continuous distributions, we use the following value to estimate
        alpha (see Clauset et al. (2009) [1], Eq. 3.1). In the limit as N goes to infinity,
        this becomes exact, and generally is a very good estimation even
        at modest values (~1000) of N.

            $$ \alpha_0 = 1 + N / ( \sum \log (x / x_{min})) $$

        For the discrete case, there is no form that is exact in the large
        N limit, but the following value is a good approximation (see ref
        [1], Eq. 3.7).

            $$ \alpha_0 = 1 + N / ( \sum \log (x / (x_{min} - 1/2))) $$

        As with the exponential distribution, lambda is taken as the inverse
        of the mean of the data:

            $$ \lambda_0 = 1 / mean(x) $$


        Parameters
        ----------
        data : array_like
            The data to use to generate the initial values of parameters.

            Should already be trimmed to the data range defined by
            `xmin` and `xmax` (if included).

        Returns
        -------
        params : dict
            A dictionary of the parameters and their values.

        References
        ----------
        [1] Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).
        Power-law distributions in empirical data. SIAM Review, 51(4),
        661–703. https://doi.org/10.1137/070710111

        """
        params = {}

        # This is generally a very good approximation of the power law
        # exponent for alpha > 1. For values of alpha < 1, it will only
        # approach 1, as expected from the 1 + ...

        n = len(data)

        # If we have a discrete distribution (ie only takes on integer
        # values) we have to shift slightly.
        if self.discrete and self.estimate_discrete and not self.xmax:
            params["alpha"] = 1 + n / np.sum(np.log(data / (self.xmin - 0.5)))

        else:
            # For continuous, we just have the usual expression.
            params["alpha"] = 1 + n / np.sum(np.log(data / (self.xmin)))

        params["Lambda"] = 1 / np.mean(data)

        return params


    def _cdf_base_function(self, x):
        from mpmath import gammainc
        from numpy import vectorize
        gammainc = vectorize(gammainc)

        CDF = ( (gammainc(1-self.alpha,self.Lambda*x)).astype('float') /
                self.Lambda**(1-self.alpha)
                    )
        CDF = 1 -CDF
        return CDF


    def _pdf_base_function(self, x):
        from numpy import exp
        return x**(-self.alpha) * exp(-self.Lambda * x)


    @property
    def _pdf_continuous_normalizer(self):
        from mpmath import gammainc
        C = ( self.Lambda**(1-self.alpha) /
                float(gammainc(1-self.alpha,self.Lambda*self.xmin)))
        return C


    @property
    def _pdf_discrete_normalizer(self):
        if 0:
            return False
        from mpmath import lerchphi
        from mpmath import exp # faster /here/ than numpy.exp
        C = ( float(exp(self.xmin * self.Lambda) /
            lerchphi(exp(-self.Lambda), self.alpha, self.xmin)) )
        if self.xmax:
            Cxmax = ( float(exp(self.xmax * self.Lambda) /
                lerchphi(exp(-self.Lambda), self.alpha, self.xmax)) )
            C = 1.0/C - 1.0/Cxmax
            C = 1.0/C
        return C


    # Deprecated
#    def pdf(self, data=None):
#        if data is None and self.parent_Fit:
#            data = self.parent_Fit.data
#
#        if not self.discrete and self.in_range() and False:
#            data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
#            from numpy import exp
#            from mpmath import gammainc
#        likelihoods = (data**-alpha)*exp(-Lambda*data)*\
#                (Lambda**(1-alpha))/\
#                float(gammainc(1-alpha,Lambda*xmin))
#            likelihoods = ( self.Lambda**(1-self.alpha) /
#                    (data**self.alpha *
#                            exp(self.Lambda*data) *
#                            gammainc(1-self.alpha,self.Lambda*self.xmin)
#                            ).astype(float)
#                    )
#            #Simplified so as not to throw a nan from infs being divided by each other
#            from sys import float_info
#            likelihoods[likelihoods==0] = 10**float_info.min_10_exp
#        else:
#            likelihoods = Distribution.pdf(self, data)
#        return likelihoods


    def _generate_random_continuous(self, r):
        def helper(r):
            from numpy import log
            from numpy.random import rand
            while 1:
                x = self.xmin - (1/self.Lambda) * log(1-r)
                p = ( x/self.xmin )**-self.alpha
                if rand()<p:
                    return x
                r = rand()
        from numpy import array
        return array(list(map(helper, r)))


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

    def pdf(self, data=None):
        """
        Returns the probability density function (normalized histogram) of the
        theoretical distribution for the values in data within xmin and xmax,
        if present.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.

        Returns
        -------
        probabilities : array
        """
        if data is None and self.parent_Fit:
            data = self.parent_Fit.data

        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        n = len(data)
        from sys import float_info
        from numpy import tile
        if not self.in_range():
            return tile(10**float_info.min_10_exp, n)

        if not self.discrete:
            f = self._pdf_base_function(data)
            C = self._pdf_continuous_normalizer
            if C > 0:
                likelihoods = f/C
            else:
                likelihoods = tile(10**float_info.min_10_exp, n)
        else:
            if self._pdf_discrete_normalizer:
                f = self._pdf_base_function(data)
                C = self._pdf_discrete_normalizer
                likelihoods = f*C
            elif self.discrete_approximation=='round':
                likelihoods = self._round_discrete_approx(data)
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

    def _round_discrete_approx(self, data):
        """
        This function reformulates the calculation to avoid underflow errors
        with the erf function. As implemented, erf(x) quickly approaches 1
        while erfc(x) is more accurate. Since erfc(x) = 1 - erf(x),
        calculations can be written using erfc(x)
        """
        import numpy as np
        import scipy.special as ss
        """ Temporarily expand xmin and xmax to be able to grab the extra bit of
        probability mass beyond the (integer) values of xmin and xmax
        Note this is a design decision. One could also say this extra
        probability "off the edge" of the distribution shouldn't be included,
        and that implementation is retained below, commented out. Note, however,
        that such a cliff means values right at xmin and xmax have half the width to
        grab probability from, and thus are lower probability than they would otherwise
        be. This is particularly concerning for values at xmin, which are typically
        the most likely and greatly influence the distribution's fit.
        """
        lower_data = data-.5
        upper_data = data+.5
        self.xmin -= .5
        if self.xmax:
            self.xmax += .5


        # revised calculation written to avoid underflow errors
        arg1 = (np.log(lower_data)-self.mu) / (np.sqrt(2)*self.sigma)
        arg2 = (np.log(upper_data)-self.mu) / (np.sqrt(2)*self.sigma)
        likelihoods = 0.5*(ss.erfc(arg1) - ss.erfc(arg2))
        if not self.xmax:
            norm = 0.5*ss.erfc((np.log(self.xmin)-self.mu) / (np.sqrt(2)*self.sigma))
        else:
            # may still need to be fixed
            norm = - self._cdf_xmin + self._cdf_base_function(self.xmax)
        self.xmin +=.5
        if self.xmax:
            self.xmax -= .5

        return likelihoods/norm

    def cdf(self, data=None, survival=False):
        """
        The cumulative distribution function (CDF) of the lognormal
        distribution. Calculated for the values given in data within xmin and
        xmax, if present. Calculation was reformulated to avoid underflow
        errors

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.
        survival : bool, optional
            Whether to calculate a CDF (False) or CCDF (True).
            False by default.

        Returns
        -------
        X : array
            The sorted, unique values in the data.
        probabilities : array
            The portion of the data that is less than or equal to X.
        """
        from numpy import log, sqrt
        import scipy.special as ss
        if data is None and self.parent_Fit:
            data = self.parent_Fit.data

        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
        n = len(data)
        from sys import float_info
        if not self.in_range():
            from numpy import tile
            return tile(10**float_info.min_10_exp, n)

        val_data = (log(data)-self.mu) / (sqrt(2)*self.sigma)
        val_xmin = (log(self.xmin)-self.mu) / (sqrt(2)*self.sigma)
        CDF = 0.5 * (ss.erfc(val_xmin) - ss.erfc(val_data))

        norm = 0.5 * ss.erfc(val_xmin)
        if self.xmax:
            # TO DO: Improve this line further for better numerical accuracy?
            norm = norm - (1 - self._cdf_base_function(self.xmax))

        CDF = CDF/norm

        if survival:
            CDF = 1 - CDF

        possible_numerical_error = False
        from numpy import isnan, min
        if isnan(min(CDF)):
            print("'nan' in fit cumulative distribution values.", file=sys.stderr)
            possible_numerical_error = True
        #if 0 in CDF or 1 in CDF:
        #    print("0 or 1 in fit cumulative distribution values.", file=sys.stderr)
        #    possible_numerical_error = True
        if possible_numerical_error:
            print("Likely underflow or overflow error: the optimal fit for this distribution gives values that are so extreme that we lack the numerical precision to calculate them.", file=sys.stderr)
        return CDF

    def generate_initial_parameters(self, data):
        from numpy import mean, std, log
        logdata = log(data)
        return (mean(logdata), std(logdata))

    def _in_standard_parameter_range(self):
#The standard deviation can't be negative
        return self.sigma>0

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
        from mpmath import erfc
#        from scipy.special import erfc
        from scipy.constants import pi
        from numpy import sqrt, log
        C = (erfc((log(self.xmin) - self.mu) / (sqrt(2) * self.sigma)) /
             sqrt(2/(pi*self.sigma**2)))
        return float(C)

    @property
    def _pdf_discrete_normalizer(self):
        return False

    def _generate_random_continuous(self, r):
        from numpy import exp, sqrt, log, frompyfunc
        from mpmath import erf, erfinv
        #This is a long, complicated function broken into parts.
        #We use mpmath to maintain numerical accuracy as we run through
        #erf and erfinv, until we get to more sane numbers. Thanks to
        #Wolfram Alpha for producing the appropriate inverse of the CCDF
        #for me, which is what we need to calculate these things.
        erfinv = frompyfunc(erfinv,1,1)
        Q = erf( ( log(self.xmin) - self.mu ) / (sqrt(2)*self.sigma))
        Q = Q*r - r + 1.0
        Q = erfinv(Q).astype('float')
        return exp(self.mu + sqrt(2)*self.sigma*Q)

#    def _generate_random_continuous(self, r1, r2=None):
#        from numpy import log, sqrt, exp, sin, cos
#        from scipy.constants import pi
#        if r2==None:
#            from numpy.random import rand
#            r2 = rand(len(r1))
#            r2_provided = False
#        else:
#            r2_provided = True
#
#        rho = sqrt(-2.0 * self.sigma**2.0 * log(1-r1))
#        theta = 2.0 * pi * r2
#        x1 = exp(rho * sin(theta))
#        x2 = exp(rho * cos(theta))
#
#        if r2_provided:
#            return x1, x2
#        else:
#            return x1


class Lognormal_Positive(Lognormal):

    @property
    def name(self):
        return "lognormal_positive"

    def _in_standard_parameter_range(self):
#The standard deviation and mean can't be negative
        return (self.sigma>0 and self.mu>0)
