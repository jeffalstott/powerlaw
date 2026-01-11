"""
This file contains the functions and classes for fitting data to a
distribution.
"""
# These packages are pretty fundamental to the
# operation of this library, so we should import them at the top.
import numpy as np
from numpy import nan

import matplotlib.pyplot as plt

# For float errors
import sys
# For checking how many processes are available
import os
# For parallelization
import multiprocessing

# So we can ignore this warning while fitting xmin
from scipy.optimize import OptimizeWarning

import warnings
from tqdm import tqdm

from .plotting import *
from .statistics import *
from .distributions import *

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

SUPPORTED_DISTRIBUTION_LIST = list(SUPPORTED_DISTRIBUTIONS.keys())

"""
Currently just templated; doesn't work yet.

Whether to enable parallelization for certain heavy calculations, eg. 
fitting the xmin value.
"""
PARALLEL_ENABLE = False
"""
Currently just templated; doesn't work yet.

This is the number of cores that the library should leave free when doing
certain heavy calculations. For example, if you have 8 cores, and this is
set to 2, then the processing would use (up to) 6 cores.
"""
PARALLEL_UNUSED_CORES = 2


class Fit(object):
    """
    A class to manage data and fits to various distributions.

    For fits to power laws, the methods of Clauset et al. 2007 are used.
    These methods identify the portion of the tail of the distribution that
    follows a power law, beyond a value ``xmin``. If no ``xmin`` is
    provided, the optimal one is calculated and assigned at initialization.

    All supported distributions can be accessed as properties of this
    class, at which time they will be automatically fit to the data.

    Parameters
    ----------
    data : array_like
        The data to fit.

    discrete : bool, optional
        Whether the data is discrete (integers).

    xmin : int or float, optional
        The data value beyond which distributions should be fitted. If
        ``None`` an optimal one will be calculated.

    xmax : int or float, optional
        The maximum value of the fitted distributions.

    estimate_discrete : bool, optional
        Whether to estimate the fit of a discrete power law using fast
        analytical methods, instead of calculating the fit exactly with
        slow numerical methods. Most accurate when ``xmin`` > 6.

    discrete_normalization : {"round", "sum"}, optional
        Approximation method to use in calculating the PDF (especially the
        PDF normalization constant) for a discrete distribution in the case
        that there is no analytical expression available.

        ``"round"`` uses the probability mass in the range ``[x - 0.5, x + 0.5]``
        for each data point.

        ``"sum"`` simply sums the PDF over the defined range to compute the
        normalization.

    sigma_threshold : float, optional
        Upper limit on the standard error of the power law fit. Used after
        fitting, when identifying valid ``xmin`` values.

    initial_parameters : dict, optional
        The initial parameters for fitting various distributions.

        Must be given as a dictionary so it is clear which value corresponds
        to which parameter of different distributions.

    parameter_ranges : dict, optional
        Dictionary of valid parameter ranges for fitting. Formatted as a
        dictionary of parameter names (eg. ``'alpha'``) and tuples/lists/etc.
        of their lower and upper limits (eg. ``(1.5, 2.5)``, ``(None, .1)``).

        The use of ``None`` is preferred over ``np.inf`` to indicate an
        unbounded limit.

    parameter_constraints : dict or list of dict
        Constraints amongst parameters during fitting. Constraint function(s)
        should take the distribution object as the only argument.
        The return value of the function should be some numerical value
        which is either 0 when the constraint is satisfied if an equality
        constraint, or a positive non-zero value when the constraint is
        satisfied if an inequality constraint.

        For example, if I want to enforce that ``param1`` is greater than
        ``param2``, I would define my function:

        .. code-block::

            def constraint(dist):
                return dist.param1 - dist.param2

            constraint_dict = {"type": 'ineq',
                               "fun": constraint}

        The dictionary (or dictionaries) should contain only the 
        following keys:

        ``"type"``: The type of the constraint, either ``"eq"`` or
        ``"ineq"``.

        ``"fun"``": The function that implements the constraint.

        ``"dists"``: The name of the distributions that this constraint
        applies to. If not provided, constraint will be applied to
        all distributions.

        After some processsing and wrapping, these constraints are
        eventually sent to ``scipy.optimize.minimize(constraints=...)``;
        see their documentation for more information.

    xmin_distance : {'D', 'V', 'Asquare'}, optional
        The distance metric used to determine which value of xmin
        gives the best fit.

        ``'D'`` is Kolmogorov-Smirnov, ``'V'`` is Kuiper, ``'Asquare'`` is Anderson-
        Darling. For more information on these, see the documentation
        for ``Distribution.compute_distance_metrics()``.

    xmin_distribution : {'power_law'}, optional
        The distribution to use in finding the optimal ``xmin`` value.

        Currently, only ``'power_law'`` is supported.

    verbose: {0, 1, 2} or bool, optional
        Whether to print updates about where we are in the fitting process.
        
        ``0`` or ``False`` means print nothing, ``1`` or ``True`` means
        only print warnings (default), ``2`` means print status messages
        and warnings.
    
    """
    def __init__(self,
                 data,
                 discrete=False,
                 xmin=None,
                 xmax=None,
                 fit_method='likelihood',
                 estimate_discrete=None,
                 discrete_normalization='round',
                 sigma_threshold=None,
                 initial_parameters=None,
                 parameter_ranges=None,
                 parameter_constraints=None,
                 xmin_distance='D',
                 xmin_distribution='power_law',
                 verbose=1):

        self.verbose = verbose

        self.data_original = data
        self.data = np.asarray(self.data_original, dtype='float')

        if self.data.ndim != 1:
            raise ValueError("Input data must be one-dimensional")

        self.discrete = discrete

        self.fit_method = fit_method
        self.estimate_discrete = estimate_discrete
        self.discrete_normalization = discrete_normalization
        self.sigma_threshold = sigma_threshold

        self.initial_parameters = initial_parameters
        self.parameter_ranges = parameter_ranges
        self.parameter_constraints = parameter_constraints

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

        if 0 in self.data:
            if self.verbose:
                warnings.warn("Values less than or equal to 0 in data. Throwing out 0 or negative values.")

            self.data = self.data[self.data > 0]

        # Sort the data
        if not all(self.data[i] <= self.data[i+1] for i in range(len(self.data)-1)):
            self.data = np.sort(self.data)

        self.fitting_cdf_bins, self.fitting_cdf = cdf(self.data, xmin=None, xmax=self.xmax)

        # No need to define this again, as we can just copy it from the
        # static variable.
        self.supported_distributions = SUPPORTED_DISTRIBUTIONS

        self.xmin_distribution_cls: type[Distribution] = self.supported_distributions[xmin_distribution]

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
        before they exist. That being said, this still only works after the
        Fit object is created.
        """
        current_attrs = self.__dict__.keys()
        total_attrs = list(current_attrs) + list(self.supported_distributions)

        # Take unique values and cast to tuple
        return tuple(np.unique(total_attrs))


    def __getattr__(self, name):
        """
        This function is redefined such that we can access the supported
        distributions and have them inherit the data and options from the
        Fit class.
        """
        # This is used for getting the individual distributions, which we
        # only fit if they are accessed.
        if name in SUPPORTED_DISTRIBUTION_LIST:
            dist = self.supported_distributions[name]

            # Create the distribution and set it
            # Note: This recomputes for every access, so you should always
            # save the property externally if you want to reference back.
            setattr(self,
                    name,
                    dist(data=self.data,
                         xmin=self.xmin,
                         xmax=self.xmax,
                         discrete=self.discrete,
                         fit_method=self.fit_method,
                         estimate_discrete=self.estimate_discrete,
                         discrete_normalization=self.discrete_normalization,
                         parameters=self.initial_parameters,
                         parameter_ranges=self.parameter_ranges,
                         parameter_constraints=self.parameter_constraints,
                         parent_Fit=self))

            return getattr(self, name)

        else:
            raise AttributeError(name)

    @property
    def xmin_distribution(self):
        return getattr(self, self.xmin_distribution_cls.name)

    def find_xmin(self, xmin_distance=None):
        """
        Returns the optimal xmin beyond which the scaling regime of the power
        law fits best. The attribute ``self.xmin`` of the Fit object is also set.

        The optimal ``xmin`` beyond which the scaling regime of the power law fits
        best is identified by minimizing the Kolmogorov-Smirnov distance
        between the data and the theoretical power law fit.
        This is the method of Clauset et al. 2007.

        Much of the rest of this function was inspired by Adam Ginsburg's
        plfit code, specifically the mapping and sigma threshold behavior:

        http://code.google.com/p/agpy/source/browse/trunk/plfit/plfit.py?spec=svn359&r=357

        Parameters
        ----------
        xmin_distance : {'D', 'V', 'Asquare'}, optional
            The distance metric used to determine which value of xmin
            gives the best fit.

            ``'D'`` is Kolmogorov-Smirnov, ``'V'`` is Kuiper, ``'Asquare'`` is Anderson-
            Darling. For more information on these, see the documentation
            for ``Distribution.compute_distance_metrics()``.

        Returns
        -------

        xmin : float
            The optimal xmin value for the fit.
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

        # Originally, we just used every single datapoint as a possible
        # xmin value, but this probably *way* oversamples the values we
        # actually need to test. An alternative, which is much faster, is
        # to just generate evenly spaced values.

        # 10% of the number of datapoints sounds good. And note that we
        # only generate bins up into the 3rd to last point so we always
        # have enough points to calculate distance metrics.
        # DEBUG
        #max_bin_value = np.sort(possible_xmin)[-3]
        #possible_xmin = np.logspace(np.log10(np.min(self.data)), np.log10(max_bin_value), len(self.data) // 10)[:-5]

        # If not provided here, take the value from the constructor
        if xmin_distance is None:
            xmin_distance = self.xmin_distance

        if len(possible_xmin) < 2:
            warnings.warn("Less than 2 unique data values for fitting xmin! Returning nans.")
            self.xmin = nan
            self.D = nan
            self.V = nan
            self.Asquare = nan
            self.Kappa = nan
            self.n_tail = nan
            setattr(self, xmin_distance+'s', np.array([nan]))
            self.noise_flag = True

            return self.xmin


        num_xmin = len(possible_xmin)
        # The original documentation states that it is desired to hold onto
        # the xmin fitting data (below) because the user may want to
        # explore if there are multiple possible fits for a single dataset.
        # That being said, I think it is a little confusing to have these
        # variables directly available as properties of this class. I
        # propose a better alternative is to have them contained in a
        # dictionary called xmin_fitting_results.
        distances = np.zeros(num_xmin)
        param_names = self.xmin_distribution_cls.parameter_names
        num_params = len(param_names)
        params = np.zeros((num_params, num_xmin))
        # Used to be called in_ranges
        valid_fits = np.zeros(num_xmin, dtype=bool)

        def fit_function(xmin):

            # Generate a distribution with the current values of xmin
            dist = self.xmin_distribution_cls(xmin=xmin,
                                        xmax=self.xmax,
                                        discrete=self.discrete,
                                        fit_method=self.fit_method,
                                        data=self.data,
                                        parameters=self.initial_parameters,
                                        parameter_ranges=self.parameter_ranges,
                                        parameter_constraints=self.parameter_constraints,
                                        parent_Fit=self,
                                        estimate_discrete=self.estimate_discrete,
                                        verbose=0)

            params = [getattr(dist, param) for param in param_names]
            valid = dist.in_range() and not dist.noise_flag
            return getattr(dist, xmin_distance), valid, params

        # Disable all warnings so we don't get messages since we'll be
        # fitting a lot of times. Otherwise, you'll almost always get an
        # OptimizeWarning from scipy, especially towards the higher values
        # of xmin.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            # TODO parallelize
            # This is slightly harder than I thought it would be since I can't
            # directly parallelize a function that isn't defined at the top
            # level. The alternative is to use a third-party library like
            # multiprocess but this is something to ask.
            if PARALLEL_ENABLE:
                raise NotImplementedError('Parallelization not yet implemented! Use `PARALLEL_ENABLE=False`')

                # See the os documentation on for this below; note that newer
                # versions of python (3.13+) have a function `process_cpu_count()`
                # that does this in a cleaner way, but it's probably better
                # to be backwards compatible.
                # https://docs.python.org/3.11/library/os.html#os.cpu_count
                usable_cores = len(os.sched_getaffinity(0)) - PARALLEL_UNUSED_CORES
                usable_cores = max(usable_cores, 1)

                # Create a pool of workers
                with multiprocessing.Pool(usable_cores) as pool:
                    # We don't need the xmin to be tested in order, so we
                    # use an unordered map
                    result_mapping = pool.imap_unordered(fit_function, possible_xmin)
                    for result in tqdm(result_mapping, desc="Fitting xmin") if self.verbose else result_mapping:
                        distances[i], valid_fits[i], params[:, i] = result

            else:
                # For non-parallel case, we just use a simple for loop
                for i in tqdm(range(num_xmin), desc='Fitting xmin') if self.verbose else range(num_xmin):
                    distances[i], valid_fits[i], params[:, i] = fit_function(possible_xmin[i])
      

        # The possible xmin values should of course have all parameters
        # within the proper range.
        good_indices = valid_fits

        # If we have a threshold, we throw out any values that are below
        # that
        if self.sigma_threshold and 'sigma' in param_names:
            sigma_idx = param_names.index('sigma')
            good_indices = good_indices * (params[sigma_idx] < self.sigma_threshold)

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
            warnings.warn('No valid values for xmin found.')

        # Set the Fit's xmin to the optimal xmin
        self.xmin = possible_xmin[min_index]
        setattr(self, xmin_distance, distances[min_index])

        # Save the fitting information to a dictionary
        xmin_fitting_results = {"distances": distances,
                                "xmins": possible_xmin,
                                "min_index":min_index,
                                "valid_fits": valid_fits}
        for vals, name in zip(params, param_names):
            xmin_fitting_results[name] = vals
        self.xmin_fitting_results = xmin_fitting_results

        # Update the fitting CDF given the new xmin, in case other objects, like
        # Distributions, want to use it for fitting (like if they do KS fitting)
        self.fitting_cdf_bins, self.fitting_cdf = self.cdf()

        return self.xmin


    # There used to be a nested_distribution_compare but it is redundant
    # since we can just use the nested keyword of distribution_compare.


    def distribution_compare(self, dist1, dist2, nested=None, **kwargs):
        """
        Returns the loglikelihood ratio, and its p-value, between the two
        distribution fits, assuming the candidate distributions are nested.

        Parameters
        ----------
        dist1 : str
            Name of the first candidate distribution (eg. 'power_law')

        dist2 : str
            Name of the second candidate distribution (eg. 'exponential')

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
            Significance of R.
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
        Another name for ``distribution_compare``.

        Parameters
        ----------
        dist1 : str
            Name of the first candidate distribution (eg. 'power_law')

        dist2 : str
            Name of the second candidate distribution (eg. 'exponential')

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
            Significance of R.
        """
        return self.distribution_compare(dist1, dist2, nested=nested, **kwargs)


    def cdf(self, original_data=False):
        """
        Returns the cumulative distribution function of the data.

        Parameters
        ----------
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object
            (True) or only the data within the fitting range (False).

        Returns
        -------
        X : numpy.ndarray
            The sorted, unique values in the data.

        probabilities : numpy.ndarray
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

        return cdf(data, xmin=xmin, xmax=xmax)


    def ccdf(self, original_data=False):
        """
        Returns the complementary cumulative distribution function of the data.

        Parameters
        ----------
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object
            (True) or only the data within the fitting range (False).

        Returns
        -------
        X : numpy.ndarray
            The sorted, unique values in the data.

        probabilities : numpy.ndarray
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

        return ccdf(data, xmin=xmin, xmax=xmax)


    def pdf(self, original_data=False, linear_bins=False, bins=None):
        """
        Returns the probability density function (normalized histogram) of the
        data.

        Parameters
        ----------
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object
            (True) or only the data within the fitting range (False).

        linear_bins : bool, optional
            Whether to use linearly spaced bins, as opposed to logarithmically
            spaced bins (default, recommended for log-log plots).

        bins : array_like, optional
            The bins within which to compute the PDF.

            If not provided, will be generated based on the range of the data.
            By default, the bins will be logarithmically spaced, but can be
            linear if `linear_bins=True`.

        Returns
        -------
        bin_edges : numpy.ndarray
            The edges of the bins of the probability density function.

        probabilities : numpy.ndarray
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

        edges, hist = pdf(data, xmin=xmin, xmax=xmax, linear_bins=linear_bins, bins=bins)

        return edges, hist


    def plot_cdf(self, original_data=False, ax=None, **kwargs):
        """
        Plot the cumulative distribution function (CDF) to a new figure or
        to axis ``ax`` if provided.

        Parameters
        ----------
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object
            (True) or only the data within the fitting range (False).

        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.

        kwargs
            Other keyword arguments are passed to `matplotlib.pyplot.plot()`.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        if original_data:
            data = self.data_original
            xmin = None
            xmax = None

        else:
            data = self.data
            xmin = self.xmin
            xmax = self.xmax

        return plot_cdf(data, xmin=xmin, xmax=xmax, ax=ax, **kwargs)


    def plot_ccdf(self, original_data=False, ax=None, **kwargs):
        """
        Plot the complementary cumulative distribution function (CCDF) to a
        new figure or to axis ``ax`` if provided.

        Parameters
        ----------
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object
            (True) or only the data within the fitting range (False).

        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.

        kwargs
            Other keyword arguments are passed to `matplotlib.pyplot.plot()`.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        if original_data:
            data = self.data_original
            xmin = None
            xmax = None

        else:
            data = self.data
            xmin = self.xmin
            xmax = self.xmax

        return plot_ccdf(data, xmin=xmin, xmax=xmax, ax=ax, **kwargs)


    def plot_pdf(self, original_data=False, linear_bins=False, bins=None, ax=None, **kwargs):
        """
        Plot the probability density function (PDF) or the data to a new figure
        or to axis ``ax`` if provided.

        Parameters
        ----------
        original_data : bool, optional
            Whether to use all of the data initially passed to the Fit object
            (True) or only the data within the fitting range (False).

        linear_bins : bool, optional
            Whether to use linearly spaced bins, as opposed to logarithmically
            spaced bins (recommended for log-log plots).

        bins : array_like, optional
            The bins within which to compute the PDF.

            If not provided, will be generated based on the range of the data.
            By default, the bins will be logarithmically spaced, but can be
            linear if ``linear_bins=True``.

        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.

        kwargs
            Other keyword arguments are passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        if original_data:
            data = self.data_original
            xmin = None
            xmax = None

        else:
            data = self.data
            xmin = self.xmin
            xmax = self.xmax

        return plot_pdf(data, xmin=xmin, xmax=xmax, linear_bins=linear_bins, bins=bins, ax=ax, **kwargs)


