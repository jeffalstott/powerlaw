"""
This file contains the functions and classes for fitting data to a
distribution.
"""
# These packages are pretty fundamental to the
# operation of this library, so we should import them at the top.
import numpy as np
from numpy import nan

import matplotlib.pyplot as plt

import sys
import warnings
from tqdm import tqdm

from .plotting import *
from .statistics import *
from .distributions import *

# TODO List
# Change name of error from sigma since sigma is already the name of a
# paremeter in lognormal distributions.

# This needs to be a list of the keys in the supported_distributions
# attribute of the Fit class.  The __getattr__ method needs the list.
# If it uses supported_distributions.keys(), then it gets into an
# infinte loop when unpickling a Fit object.  Hence the need for a
# separate list outside the scope of the Fit class.
SUPPORTED_DISTRIBUTIONS = {'power_law': Power_Law,
                           'lognormal': Lognormal,
                           'exponential': Exponential,
                           'truncated_power_law': Truncated_Power_Law,
                           'stretched_exponential': Stretched_Exponential,
                           'lognormal_positive': Lognormal_Positive,
                           }
                            #'gamma': None}

SUPPORTED_DISTRIBUTION_LIST = list(SUPPORTED_DISTRIBUTIONS.keys())

"""
Whether to enable parallelization for certain heavy calculations, eg. 
fitting the xmin value.
"""
PARALLEL_ENABLE = True
"""
This is the number of cores that the library should leave free when doing
certain heavy calculations. For example, if you have 8 cores, and this is
set to 2, then the processing would use (up to) 6 cores.
"""
PARALLEL_UNUSED_CORES = 2


class Fit(object):
    """
    A fit of a data set to various probability distributions, namely power
    laws. For fits to power laws, the methods of Clauset et al. 2007 are used.
    These methods identify the portion of the tail of the distribution that
    follows a power law, beyond a value xmin. If no xmin is
    provided, the optimal one is calculated and assigned at initialization.

    Parameters
    ----------
    data : list or array

    discrete : boolean, optional
        Whether the data is discrete (integers).

    xmin : int or float, optional
        The data value beyond which distributions should be fitted. If
        None an optimal one will be calculated.

    xmax : int or float, optional
        The maximum value of the fitted distributions.

    verbose: bool, optional
        Whether to print updates about where we are in the fitting process.
        Default False.

    estimate_discrete : bool, optional
        Whether to estimate the fit of a discrete power law using fast
        analytical methods, instead of calculating the fit exactly with
        slow numerical methods. Very accurate with xmin>6

    sigma_threshold : float, optional
        Upper limit on the standard error of the power law fit. Used after
        fitting, when identifying valid xmin values.

    parameter_ranges : dict, optional
        Dictionary of valid parameter ranges for fitting. Formatted as a
        dictionary of parameter names ('alpha' and/or 'sigma') and tuples
        of their lower and upper limits (ex. (1.5, 2.5), (None, .1)

    pdf_ends_at_xmax: bool, optional
        Whether to use the pdf that has an upper cutoff at xmax to fit the 
        powerlaw distribution. 
    """

    def __init__(self, data,
                 discrete=False,
                 xmin=None, xmax=None,
                 verbose=False,
                 fit_method='likelihood',
                 estimate_discrete=True,
                 discrete_approximation='round',
                 sigma_threshold=None,
                 parameter_ranges=None,
                 fit_optimizer=None,
                 xmin_distance='D',
                 xmin_distribution='power_law',
                 pdf_ends_at_xmax=False,
                 **kwargs):

        self.verbose = verbose

        self.data_original = data
        self.data = np.asarray(self.data_original, dtype='float')

        if self.data.ndim != 1:
            raise ValueError("Input data must be one-dimensional")

        self.discrete = discrete

        self.fit_method = fit_method
        self.estimate_discrete = estimate_discrete
        self.discrete_approximation = discrete_approximation
        self.sigma_threshold = sigma_threshold
        self.parameter_ranges = parameter_ranges

        # We keep track of the xmin and xmax values if they are provided.
        # I don't really see the purpose for this variable, but I'll leave
        # it in for backwards compatability.
        self.fixed_xmax = (xmax is not None)
        self.xmax = xmax

        # Trim the data above xmax
        if self.fixed_xmax:
            self.xmax = float(self.xmax)

            # We keep track of this to compute n_tail at the end of __init__
            n_above_max = sum(self.data > self.xmax)

            self.data = self.data[self.data <= self.xmax]

        else:
            n_above_max = 0

        # We also need to separately note if xmin was given since we will
        # eventually fit it if it isn't provided, so we need to be able
        # to tell if that value was given or fitted.
        # If xmin has __iter__, that means it's a list/tuple/array
        # Note that originally `fixed_xmin` was defined later in the code,
        # but we can easily define it now so we may as well.

        # If we have a range, we assign that
        if hasattr(xmin, '__iter__'):
            assert len(xmin) == 2, f'Invalid range given for xmin ({xmin}), should be iterable of length 2'

            self.xmin = None
            self.xmin_range = xmin
            self.fixed_xmin = False

        elif (xmin is not None):
            # If we just have a value, we have a fixed value and don't have
            # to fit anything
            self.xmin = xmin
            self.xmin_range = (None, None)
            self.fixed_xmin = True

        else:
            # Otherwise, we have no fixed xmin and our range is the entire
            # span of the data.
            self.xmin = None
            self.xmin_range = (np.min(self.data), np.max(self.data))
            self.fixed_xmin = False

        self.xmin_distance = xmin_distance
        self.pdf_ends_at_xmax = pdf_ends_at_xmax

        if 0 in self.data:
            if self.verbose:
                warnings.warn("Values less than or equal to 0 in data. Throwing out 0 or negative values")

            self.data = self.data[self.data > 0]

        # Sort the data
        if not all(self.data[i] <= self.data[i+1] for i in range(len(self.data)-1)):
            self.data = np.sort(self.data)

        self.fitting_cdf_bins, self.fitting_cdf = cdf(self.data, xmin=None, xmax=self.xmax)

        # No need to define this again, as we can just copy it from the
        # static variable.
        self.supported_distributions = SUPPORTED_DISTRIBUTIONS

        self.xmin_distribution = self.supported_distributions[xmin_distribution]
        self.xmin_distribution.pdf_ends_at_xmax = self.pdf_ends_at_xmax

        # If we have a fixed xmin, we can directly fit a power law distribution
        if self.fixed_xmin:
            self.xmin = float(xmin)

        else:
            if self.verbose:
                print(f'Calculating best minimal value for {xmin_distribution.replace("_"," ")} fit')

            # This function tries to optimize the fit based on the xmin
            self.find_xmin()

        # Crop the data to the xmin and
        self.data = self.data[self.data >= self.xmin]
        self.n = float(len(self.data))
        self.n_tail = self.n + n_above_max


    def __dir__(self):
        """
        In general, we don't do any actual fitting until a specific
        distribution is called, like `fit.power_law`. This means we
        normally can't autocomplete the distribution names since they
        don't exist until we actually call them.

        As such, we add the list of distribution names to the __dir__
        function used for autocomplete so we can autocomplete them
        before they exist.
        """
        current_attrs = self.__dict__.keys()
        total_attrs = list(current_attrs) + list(self.supported_distributions)

        # Take unique values and cast to tuple
        return tuple(np.unique(total_attrs))


    def __getattr__(self, name):
        # This is used for getting the individual distributions, which we
        # only fit if they are accessed.
        if name in SUPPORTED_DISTRIBUTION_LIST:
            dist = self.supported_distributions[name]

            # Create the distribution and set it
            # TODO: This recomputes for every access, not sure if that is
            # ideal.
            setattr(self,
                    name,
                    dist(data=self.data,
                         xmin=self.xmin,
                         xmax=self.xmax,
                         discrete=self.discrete,
                         fit_method=self.fit_method,
                         estimate_discrete=self.estimate_discrete,
                         discrete_approximation=self.discrete_approximation,
                         parameter_ranges=self.parameter_ranges,
                         parent_Fit=self))

            return getattr(self, name)

        else:
            raise AttributeError(name)


    def find_xmin_old(self, xmin_distance=None):
        """
        OLD VERSION FOR DEBUG
        Returns the optimal xmin beyond which the scaling regime of the power
        law fits best. The attribute self.xmin of the Fit object is also set.

        The optimal xmin beyond which the scaling regime of the power law fits
        best is identified by minimizing the Kolmogorov-Smirnov distance
        between the data and the theoretical power law fit.
        This is the method of Clauset et al. 2007.
        """
        import sys
        from numpy import unique, asarray, argmin, nan, repeat, arange
        self.given_xmin = self.fixed_xmin
        self.parameter_range = self.parameter_ranges

#Much of the rest of this function was inspired by Adam Ginsburg's plfit code,
#specifically the mapping and sigma threshold behavior:
#http://code.google.com/p/agpy/source/browse/trunk/plfit/plfit.py?spec=svn359&r=357
        if not self.given_xmin:
            possible_xmins = self.data
        else:
            possible_ind = min(self.given_xmin)<=self.data
            possible_ind *= self.data<=max(self.given_xmin)
            possible_xmins = self.data[possible_ind]
        xmins, xmin_indices = unique(possible_xmins, return_index=True)
#Don't look at last xmin, as that's also the xmax, and we want to at least have TWO points to fit!
        xmins = xmins[:-1]
        xmin_indices = xmin_indices[:-1]

        if xmin_distance is None:
            xmin_distance = self.xmin_distance

        if len(xmins)<=0:
            print("Less than 2 unique data values left after xmin and xmax "
                  "options! Cannot fit. Returning nans.", file=sys.stderr)
            from numpy import nan, array
            self.xmin = nan
            self.D = nan
            self.V = nan
            self.Asquare = nan
            self.Kappa = nan
            self.alpha = nan
            self.sigma = nan
            self.n_tail = nan
            setattr(self, xmin_distance+'s', array([nan]))
            self.alphas = array([nan])
            self.sigmas = array([nan])
            self.in_ranges = array([nan])
            self.xmins = array([nan])
            self.noise_flag = True
            return self.xmin

        def fit_function(xmin, idx, num_xmins):
            if sys.stdout.isatty():
                print('xmin progress: {:02d}%'.format(int(idx/num_xmins * 100)), end='\r')
            pl = self.xmin_distribution(xmin=xmin,
                           xmax=self.xmax,
                           discrete=self.discrete,
                           estimate_discrete=self.estimate_discrete,
                           fit_method=self.fit_method,
                           data=self.data,
                           parameter_range=self.parameter_range,
                           parent_Fit=self,
                           pdf_ends_at_xmax=self.pdf_ends_at_xmax)
            if not hasattr(pl, 'sigma'):
                pl.sigma = nan
            if not hasattr(pl, 'alpha'):
                pl.alpha = nan
            return getattr(pl, xmin_distance), pl.alpha, pl.sigma, pl.in_range()

        num_xmins = len(xmins)
        fits = asarray(list(map(fit_function, xmins, arange(num_xmins), repeat(num_xmins, num_xmins))))
        # logging.warning(fits.shape)
        setattr(self, xmin_distance+'s', fits[:,0])
        self.alphas = fits[:,1]
        self.sigmas = fits[:,2]
        self.in_ranges = fits[:,3].astype(bool)
        self.xmins = xmins

        good_values = self.in_ranges

        if self.sigma_threshold:
            good_values = good_values * (self.sigmas < self.sigma_threshold)

        if good_values.all():
            min_D_index = argmin(getattr(self, xmin_distance+'s'))
            self.noise_flag = False
        elif not good_values.any():
            min_D_index = argmin(getattr(self, xmin_distance+'s'))
            self.noise_flag = True
        else:
            from numpy.ma import masked_array
            masked_Ds = masked_array(getattr(self, xmin_distance+'s'), mask=~good_values)
            min_D_index = masked_Ds.argmin()
            self.noise_flag = False

        if self.noise_flag:
            print("No valid fits found.", file=sys.stderr)

        #Set the Fit's xmin to the optimal xmin
        self.xmin = xmins[min_D_index]
        setattr(self, xmin_distance, getattr(self, xmin_distance+'s')[min_D_index])
        self.alpha = self.alphas[min_D_index]
        self.sigma = self.sigmas[min_D_index]

        #Update the fitting CDF given the new xmin, in case other objects, like
        #Distributions, want to use it for fitting (like if they do KS fitting)
        self.fitting_cdf_bins, self.fitting_cdf = self.cdf()

        return self.xmin

    def find_xmin(self, xmin_distance=None):
        """
        Returns the optimal xmin beyond which the scaling regime of the power
        law fits best. The attribute self.xmin of the Fit object is also set.

        The optimal xmin beyond which the scaling regime of the power law fits
        best is identified by minimizing the Kolmogorov-Smirnov distance
        between the data and the theoretical power law fit.
        This is the method of Clauset et al. 2007.

        Much of the rest of this function was inspired by Adam Ginsburg's
        plfit code, specifically the mapping and sigma threshold behavior:

        http://code.google.com/p/agpy/source/browse/trunk/plfit/plfit.py?spec=svn359&r=357
        """
        # This function will be called if xmin is None, and we will already
        # have a defined xmin_range from __init__

        # Grab the indices of possible xmin values
        possible_ind = np.where((self.data >= np.min(self.xmin_range)) & (self.data < np.max(self.xmin_range)))
        possible_xmin = self.data[possible_ind]

        # Take unique values
        possible_xmin, possible_ind = np.unique(possible_xmin, return_index=True)

        # Don't look at last xmin, as that's also the xmax
        possible_xmin = possible_xmin[:-1]
        possible_ind = possible_ind[:-1]

        # If not provided here, take the value from the constructor
        if xmin_distance is None:
            xmin_distance = self.xmin_distance

        if len(possible_xmin) < 2:
            print("Less than 2 unique data values left after xmin and xmax "
                  "options! Cannot fit. Returning nans.", file=sys.stderr)
            self.xmin = nan
            self.D = nan
            self.V = nan
            self.Asquare = nan
            self.Kappa = nan
            self.alpha = nan
            self.sigma = nan
            self.n_tail = nan
            setattr(self, xmin_distance+'s', np.array([nan]))
            self.noise_flag = True
            return self.xmin

        def fit_function(xmin):

            # Generate a distribution with the current values of xmin
            pl = self.xmin_distribution(xmin=xmin,
                                        xmax=self.xmax,
                                        discrete=self.discrete,
                                        fit_method=self.fit_method,
                                        data=self.data,
                                        parameters=None,
                                        parameter_ranges=self.parameter_ranges,
                                        parent_Fit=self,
                                        estimate_discrete=self.estimate_discrete,
                                        pdf_ends_at_xmax=self.pdf_ends_at_xmax,
                                        verbose=0)

            # TODO not sure why the object wouldn't have these values.
            if not hasattr(pl, 'sigma'):
                pl.sigma = nan
            if not hasattr(pl, 'alpha'):
                pl.alpha = nan

            return getattr(pl, xmin_distance), pl.alpha, pl.sigma, pl.in_range()

        num_xmin = len(possible_xmin)

        # This used to be a map function but I think it's better to have
        # it explicitly written out, especially since I think it would be
        # nice to have the option for this to be parallelized.

        # I also don't see why we need to store the alphas and sigmas to
        # the class; I think they will not be used again, so it's more
        # clear for users if they aren't saved.
        distances = np.zeros(num_xmin)
        alphas = np.zeros(num_xmin)
        sigmas = np.zeros(num_xmin)
        in_ranges = np.zeros(num_xmin, dtype=bool)

        # TODO parallelize
        for i in tqdm(range(num_xmin), desc='Fitting xmin') if self.verbose else range(num_xmin):
            distances[i], alphas[i], sigmas[i], in_ranges[i] = fit_function(possible_xmin[i])
       
        # The possible xmin values should of course have all parameters
        # within the proper range.
        good_indices = in_ranges

        # If we have a threshold, we throw out any values that are below
        # that
        if self.sigma_threshold:
            good_indices = good_indices * (sigmas < self.sigma_threshold)

        # If we have no good values, the fit failed
        if not good_indices.any():
            # We still continue though
            self.noise_flag = True
            min_index = np.argmin(distances)

        else:
            # Otherwise, we take the lowest distance that is a good index
            masked_distances = np.ma.masked_array(distances, mask=~good_indices)
            min_index = masked_distances.argmin()
            self.noise_flag = False

        if self.noise_flag:
            # I've set this to be a warning as it is more in the spirit of
            # the previous code, though it's worth discussing if it should
            # instead just be an error.
            warnings.warn('No valid fit found.')

        # Set the Fit's xmin to the optimal xmin
        self.xmin = possible_xmin[min_index]
        setattr(self, xmin_distance, distances[min_index])
        self.alpha = alphas[min_index]
        self.sigma = sigmas[min_index]

        # DEBUG
        self.distances = distances
        self.alphas = alphas
        self.xmins = possible_xmin
        self.valid_fits = good_indices
        self.normalizers = sigmas

        # Update the fitting CDF given the new xmin, in case other objects, like
        # Distributions, want to use it for fitting (like if they do KS fitting)
        self.fitting_cdf_bins, self.fitting_cdf = self.cdf()

        return self.xmin


    def nested_distribution_compare(self, dist1, dist2, nested=True, **kwargs):
        """
        Returns the loglikelihood ratio, and its p-value, between the two
        distribution fits, assuming the candidate distributions are nested.

        Parameters
        ----------
        dist1 : string
            Name of the first candidate distribution (ex. 'power_law')
        dist2 : string
            Name of the second candidate distribution (ex. 'exponential')
        nested : bool or None, optional
            Whether to assume the candidate distributions are nested versions
            of each other. None assumes not unless the name of one distribution
            is a substring of the other. True by default.

        Returns
        -------
        R : float
            Loglikelihood ratio of the two distributions' fit to the data. If
            greater than 0, the first distribution is preferred. If less than
            0, the second distribution is preferred.
        p : float
            Significance of R
        """
        return self.distribution_compare(dist1, dist2, nested=nested, **kwargs)

    def distribution_compare(self, dist1, dist2, nested=None, **kwargs):
        """
        Returns the loglikelihood ratio, and its p-value, between the two
        distribution fits, assuming the candidate distributions are nested.

        Parameters
        ----------
        dist1 : string
            Name of the first candidate distribution (ex. 'power_law')
        dist2 : string
            Name of the second candidate distribution (ex. 'exponential')
        nested : bool or None, optional
            Whether to assume the candidate distributions are nested versions
            of each other. If None (default), the function will automatically
            set nested=True if one distribution name is a substring of the other
            (i.e., if either dist1 in dist2 or dist2 in dist1). Otherwise, it
            will assume nested=False.

        Returns
        -------
        R : float
            Loglikelihood ratio of the two distributions' fit to the data. If
            greater than 0, the first distribution is preferred. If less than
            0, the second distribution is preferred.
        p : float
            Significance of R
        """
        if (dist1 in dist2) or (dist2 in dist1) and nested is None:
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
        """
        Another name for distribution_compare.
        """
        return self.distribution_compare(dist1, dist2, nested=nested, **kwargs)

    def cdf(self, original_data=False, survival=False, **kwargs):
        """
        Returns the cumulative distribution function of the data.

        Parameters
        ----------
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object.
            If False, uses only the data used for the fit (within xmin and
            xmax.)
        survival : bool, optional
            Whether to return the complementary cumulative distribution
            function, 1-CDF, also known as the survival function.

        Returns
        -------
        X : array
            The sorted, unique values in the data.
        probabilities : array
            The portion of the data that is less than or equal to X.
        """
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
        """
        Returns the complementary cumulative distribution function of the data.

        Parameters
        ----------
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object.
            If False, uses only the data used for the fit (within xmin and
            xmax.)
        survival : bool, optional
            Whether to return the complementary cumulative distribution
            function, also known as the survival function, or the cumulative
            distribution function, 1-CCDF.

        Returns
        -------
        X : array
            The sorted, unique values in the data.
        probabilities : array
            The portion of the data that is greater than or equal to X.
        """
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
        """
        Returns the probability density function (normalized histogram) of the
        data.

        Parameters
        ----------
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object.
            If False, uses only the data used for the fit (within xmin and
            xmax.)

        Returns
        -------
        bin_edges : array
            The edges of the bins of the probability density function.
        probabilities : array
            The portion of the data that is within the bin. Length 1 less than
            bin_edges, as it corresponds to the spaces between them.
        """
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
        """
        Plots the CDF to a new figure or to axis ax if provided.

        Parameters
        ----------
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object.
            If False, uses only the data used for the fit (within xmin and
            xmax.)
        survival : bool, optional
            Whether to plot a CDF (False) or CCDF (True). False by default.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        if original_data:
            data = self.data_original
        else:
            data = self.data
        return plot_cdf(data, ax=ax, survival=survival, **kwargs)

    def plot_ccdf(self, ax=None, original_data=False, survival=True, **kwargs):
        """
        Plots the CCDF to a new figure or to axis ax if provided.

        Parameters
        ----------
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object.
            If False, uses only the data used for the fit (within xmin and
            xmax.)
        survival : bool, optional
            Whether to plot a CDF (False) or CCDF (True). True by default.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        if original_data:
            data = self.data_original
        else:
            data = self.data
        return plot_cdf(data, ax=ax, survival=survival, **kwargs)

    def plot_pdf(self, ax=None, original_data=False,
                 linear_bins=False, **kwargs):
        """
        Plots the probability density function (PDF) or the data to a new figure
        or to axis ax if provided.

        Parameters
        ----------
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object.
            If False, uses only the data used for the fit (within xmin and
            xmax.)
        linear_bins : bool, optional
            Whether to use linearly spaced bins (True) or logarithmically
            spaced bins (False). False by default.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        if original_data:
            data = self.data_original
        else:
            data = self.data
        return plot_pdf(data, ax=ax, linear_bins=linear_bins, **kwargs)


