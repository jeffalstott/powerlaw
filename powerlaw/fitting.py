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

# So we can hash the fit object for caching
import hashlib

# For saving and loading
import h5py
# For saving function source code when saving/loading
import inspect
import json
import textwrap

# Dill is a drop-in replacement for pickle that allows you to serialize
# local functions (among other improvements); used for saving and loading
# fit objects, and parallelization.
#import pickle
import dill as pickle

import warnings
from tqdm import tqdm

from .plotting import *
from .statistics import *
from .distributions import *

# Try and grab the current version of the package from the _version.py file.
# This is for comparing with saved/loaded Fit objects. If we can't find that
# file, we have to work without it.
try:
    from ._version import __version__
    POWERLAW_VERSION = __version__
except:
    # Raise a warning, in case the user isn't aware that the package
    # isn't properly installed
    warnings.warn('powerlaw version not found, likely because the package isn\'t installed properly. Not critical, but could affect caching of files.')
    POWERLAW_VERSION = None

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
Whether to enable parallelization for certain heavy calculations, eg. 
fitting the xmin value.
"""
_parallel_enable = False

"""
This is the number of cores/processes that the library should use.
"""
_parallel_cores = 1

"""
By default, the multiprocessing library has many limitations when it comes
to what functions can be used in a Pool or Process object. This is becuase
these require that you serialize the function using pickle, which is then
pass between processes.

Unfortunately, the standard library `pickle` doesn't have support for many
types of functions, including those not defined at the root level. The
function we want to parallelize for `find_xmin` is not defined at the
root level, so normally we would get an error along the lines of:
    Can't pickle local object 'Fit.find_xmin.<locals>.fit_function'

We can fix this by manually replacing multiprocessing's use of pickle to
`dill`'s version of it, which can handle these types of functions.
"""
# We imported dill as 'pickle', so all references to 'pickle' are actually
# dill.
pickle.Pickler.dumps, pickle.Pickler.loads = pickle.dumps, pickle.loads
multiprocessing.reduction.ForkingPickler = pickle.Pickler
multiprocessing.reduction.dump = pickle.dump

"""
The file types and extensions supported for saving and loading files.

Each key is the name of the format, and the value should be the list of
extensions that correspond to this format.
"""
SUPPORTED_SAVE_FORMATS = {"hdf5": ['h5', 'hdf5'],
                          "pickle": ['pkl', 'pickle']}

SUPPORTED_SAVE_FILE_EXTENSIONS = [ext for v in SUPPORTED_SAVE_FORMATS.values() for ext in v]

DEFAULT_SAVE_FORMAT = 'h5'
    
"""
Whether fits are cached automatically or not.

This variable should not be manually set, but rather is automatically
set to True when a valid path is given using Fit.set_cache_folder().
"""
_cache_enabled = False

"""
The path to the cache folder.

This variable should not be manually set, but rather set using
Fit.set_cache_folder().
"""
_cache_path = None

"""
The format to use when automatically caching files.

This variable should not be manually set, but rather set using
Fit.set_cache_format(). The default is hdf5.
"""
_cache_format = DEFAULT_SAVE_FORMAT


def set_parallel_cores(num_cores):
    """
    Set the number of cores to use in parallel for computing xmin.

    If a negative number, will leave that many cores open.
    """
    # See the os documentation on for this below; note that newer
    # versions of python (3.13+) have a function `process_cpu_count()`
    # that does this in a cleaner way, but it's probably better
    # to be backwards compatible.
    # https://docs.python.org/3.11/library/os.html#os.cpu_count
    total_cores = len(os.sched_getaffinity(0))

    global _parallel_enable, _parallel_cores

    if num_cores < 0:
        usable_cores = total_cores + num_cores
        usable_cores = max(usable_cores, 1)
    else:
        if num_cores > total_cores:
            raise ValueError(f'Attempted to parallelize using {num_cores} cores, but only {total_cores} are available.')

        usable_cores = num_cores

    _parallel_cores = usable_cores

    if _parallel_cores == 0 or  _parallel_cores == 1:
        _parallel_enable = False

    else:
        _parallel_enable = True


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

    test_all_xmin : bool
        Whether to test every single unique value in the dataset for fitting
        xmin (True) or generate a uniform distribution of values that spans
        the data.

    ignore_cache : bool
        Whether to ignore cached files, even if automatic cacheing is 
        enabled.

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
                 test_all_xmin=False,
                 ignore_cache=False,
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

        # For initial parameters and ranges, we need to do some standardization
        # such that we can nicely save and load Fit objects to files. This
        # is primarily relevant to the hashing, since we need to make sure
        # the type of numbers in either case is a float or None.
        if hasattr(initial_parameters, '__iter__'):
            for k, v in initial_parameters.items():
                initial_parameters[k] = float(v) if v else None

        self.initial_parameters = initial_parameters
 
        if hasattr(parameter_ranges, '__iter__'):
            for k, v in parameter_ranges.items():
                parameter_ranges[k][0] = float(v[0]) if v[0] else None
                parameter_ranges[k][1] = float(v[1]) if v[1] else None

        self.parameter_ranges = parameter_ranges

        # Make sure that we are given a list of constraints; so even if we
        # are only given a single one, added it to a list
        if type(parameter_constraints) is dict:
            self.parameter_constraints = [parameter_constraints]
        else:
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
        # We need to save this for hashing; otherwise, we never use it
        # again after passing to find_xmin().
        self.test_all_xmin = test_all_xmin

        ####################################
        # CHECK CACHE
        # Now that we have all of our variables set, we should check to see
        # if we need to look for a cached copy of this fit. Notably, this
        # happens before we fit xmin, since the point of caching is to
        # avoid having to repeat that calculation if possible.
        if _cache_enabled and not ignore_cache:
            potential_cache_file = os.path.join(_cache_path, f'{hash(self)}.{_cache_format}')

            if os.path.exists(potential_cache_file):
                if self.verbose:
                    print(f'Found cached file: {potential_cache_file}')

                # You can't just overwrite self, so we have to update
                # every variable
                loaded_fit = Fit.load(potential_cache_file)
                print(self.__dict__)
                self.__dict__.update(loaded_fit.__dict__)
                print(self.__dict__)
                return

            # If we don't find a cache file, no problem, we just continue on with
            # calculating xmin.


        ####################################
        # FIT XMIN
        # If we have a fixed xmin, we can directly fit a power law distribution
        if self.fixed_xmin:
            self.xmin = float(xmin)
        else:
            if self.verbose:
                print(f'Calculating best minimal value for {xmin_distribution.replace("_"," ")} fit')

            # This function tries to optimize the fit based on the xmin
            self.find_xmin(self.xmin_distance, self.test_all_xmin)

        # Crop the data to the xmin and
        self.data = self.data[self.data >= self.xmin]
        self.n = float(len(self.data))
        self.n_tail = self.n + n_above_max

        ####################################
        # CREATE CACHE
        if _cache_enabled and not ignore_cache:
            new_cache_file = os.path.join(_cache_path, f'{hash(self)}.{_cache_format}')

            if not os.path.exists(new_cache_file):
                if verbose:
                    print(f'Caching file at: {new_cache_file}')
                self.save(new_cache_file)



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

    def find_xmin(self, xmin_distance=None, test_all_xmin=False):
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


        # We have the option to use every value (usually quite slow) or
        # an evenly sampled set of values as possible xmin values (usually
        # much faster).
        if test_all_xmin:
            # Don't look at last xmin, as that's also the xmax
            possible_xmin = possible_xmin[:-1]
            #possible_ind = possible_ind[:-1]

        else:
            max_bin_value = np.sort(possible_xmin)[-3]

            # Use 1% of the datapoints, but always at least 100 if the 1%
            # is less than that. This should work well for most cases, but
            # might need to be improved in the future.
            num_bins = max(100, len(self.data) // 100)

            # These are logarithmically spaced
            possible_xmin = np.logspace(np.log10(np.min(self.data)), np.log10(max_bin_value), num_bins)

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

            # TODO improve parallelization
            # This currently works, but we see less time reduction than
            # I would expect. For example, I would expect that using 4
            # processes/cores would give something like a speedup of close
            # 4 (though definitely less), but it usually ends up just around
            # 2x. This means either the majority of the time is spent
            # communicating between processes (eg. writing arrays), or
            # something else is weird.

            global _parallel_enable, _parallel_cores
            if _parallel_enable:
                #raise NotImplementedError('Parallelization not yet implemented! Use `PARALLEL_ENABLE=False`')

                # Create a pool of workers
                with multiprocessing.Pool(_parallel_cores) as pool:
                    # We don't need the xmin to be tested in order, so we
                    # use an unordered map

                    # chunksize controls how many iterations a process works
                    # on before communicating back to the main thread. I
                    # experimented with this a bit, but I think we can get
                    # better results by choosing this well.
                    #chunksize = max(int(len(possible_xmin) / PARALLEL_CORES / 100), 1)
                    chunksize = 1

                    result_mapping = enumerate(pool.imap_unordered(fit_function, possible_xmin, chunksize=chunksize))
                    for i, result in tqdm(result_mapping, desc="Fitting xmin") if self.verbose else result_mapping:
                        distances[i], valid_fits[i], params[:,i] = result

            else:
                # For non-parallel case, we just use a simple for loop
                for i in tqdm(range(num_xmin), desc='Fitting xmin') if self.verbose else range(num_xmin):
                    distances[i], valid_fits[i], params[:,i] = fit_function(possible_xmin[i])
      

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

    def save(self, filename, format=None):
        """
        Save the fit object to a file.

        Note that this doesn't use Python's pickling framework, but instead
        saves the relevant data and fitting information in an hdf5 file.
        This way, the data can easily be recovered even if you aren't
        working with this library or even with Python. The cost of this is that
        the saving and loading methods are relatively complex (or at least
        just long) since we have to parse all of the important information
        into generic formats. As an end-user, this isn't a problem at all,
        but might make maintenance slightly more difficult.

        Note that saving and loading imposes some restrictions on the
        form of constraint functions. This saving and loading is done by
        using the actual source code of the constraint functions, which
        means that each function must be fully self-contained. For more
        information, see the tutorial page about parameter constraints.

        The current version of ``powerlaw`` will be saved within the file;
        if you try to load a file from a different version, you will be shown
        a warning.

        Parameters
        ----------
        filename : str or Path
            The file to which to save the ``Fit`` data.
        """

        # Determine what type of file we have
        file_extension = filename.split('.')[-1]

        # If we have no extension, we follow the value of the format kw
        if '.' not in filename or file_extension not in SUPPORTED_SAVE_FILE_EXTENSIONS:

            if format is not None and format in SUPPORTED_SAVE_FILE_EXTENSIONS:
                full_filename = filename + '.' + format

            elif format is not None:
                raise ValueError('Desired format ({format}) is unsupported.')

            else:
                # Just use the default format
                full_filename = filename + '.' + DEFAULT_SAVE_FORMAT
                format = DEFAULT_SAVE_FORMAT

        else:
            full_filename = filename
            format = file_extension

        if format in ['h5', 'hdf5']:
            _write_hdf5_file(full_filename, self)

        elif format in ['pkl', 'pickle']:
            _write_pickle_file(full_filename, self)

        else:
            raise ValueError('Unsupported save format!')


    @staticmethod
    def load(filename, verbose=1):
        """
        Load a saved fit object.

        See also ``powerlaw.Fit.save()``.

        Note that the loading and saving imposes some restrictions on the
        form of constraint functions. This saving and loading is done by
        using the actual source code of the constraint functions, which
        means that each function must be fully self-contained. For more
        information, see the tutorial page about parameter constraints.

        This function can load the following types, signified by their
        respective extensions:

            HDF5: .h5, .hdf5
            Pickle: .pkl, .pickle

        Parameters
        ----------
        filename : str or Path
            The path to the file that will be loaded.

        Returns
        -------
        fit : Fit
            The loaded fit object, which should be functionally identical
            to the saved object.
        """
        # Determine what type of file we have
        file_extension = filename.split('.')[-1]

        # HDF5
        if file_extension.lower() in ['h5', 'hdf5']:
            return _parse_hdf5_file(filename, verbose)

        # Pickle
        elif file_extension.lower() in ['pkl', 'pickle']:
            return _parse_pickle_file(filename)

        else:
            raise ValueError(f'Unknown filetype passed ({filename}); make sure that your file has an appropriate extension. See documentation for this function for acceptable extensions.')


    def __eq__(self, other):
        """
        Check for equality between two ``Fit`` objects.

        This is done using a custom implementation that checks the actual
        fit data and parameters against each other, such that two separate
        instances with the exact same information will be deemed equal.
        """
        return hash(self) == hash(other)

    
    def __hash__(self):
        """
        Generate a unique hash for this fit based on the data and other
        fitting parameters, such that the hash is the same for any two
        identical fits.

        Used for automatic caching of ``Fit`` objects.

        Note that this doesn't do any checks about specific distributions,
        eg. `fit.power_law`; this is because these may or may not exist
        depending on whether the user has accessed them.

        Returns
        -------

        hash : int
            The integer hash of the ``Fit`` object
        """
        # We cast the data to a specific type, since it could be passed
        # as a float32, float64, int32, int64, etc.
        # We choose float32 since it's not too large, but includes enough
        # precision.
        retyped_data = np.sort(np.array(self.data_original, dtype=np.float32))
        # Numpy arrays are unhashable, so we need a fixed tuple.
        retyped_data = tuple(retyped_data)

        # We need to be able to create this hash *before* computing xmin
        # (if requested), so we can't use the actual xmin value in the hashing
        # if one isn't explicitly specified.
        xmin_value = np.float32(self.xmin if self.fixed_xmin else -1)
        xmax_value = np.float32(self.xmax if self.xmax else -1)

        # For the parameter ranges and initial values, we need to dump the
        # dictionary objects with sorted keys, otherwise we could see
        # differences based on the arbitrary order in which keys are added.
        if hasattr(self.parameter_ranges, '__iter__'):
            parameter_ranges_value = json.dumps(self.parameter_ranges, sort_keys=True)
        else:
            parameter_ranges_value = None

        if hasattr(self.initial_parameters, '__iter__'):
            initial_parameters_value = json.dumps(self.initial_parameters, sort_keys=True)
        else:
            initial_parameters_value = None

        # TODO: It might be good to include parameter constraint functions
        # here too, but finding a consistent hash for a function like that
        # would be very difficult...

        # Also, we have to cast the bool variables to actual bools, since
        # for some reason they might get transformed into numpy versions
        # of True and False (numpy.True_ and numpy.False_) when being read
        # from an hdf5 file.

        # The actual object we will hash is a tuple of the important
        # identifying information
        hash_data = retyped_data + (bool(self.fixed_xmin),
                                    xmin_value,
                                    xmax_value,
                                    bool(self.discrete),
                                    self.fit_method,
                                    bool(self.estimate_discrete) if self.discrete else False,
                                    self.discrete_normalization if self.discrete else '',
                                    self.xmin_distribution_cls.name,
                                    self.xmin_distance,
                                    bool(self.test_all_xmin),
                                    parameter_ranges_value,
                                    initial_parameters_value)

        # Python's basic hash() function isn't consistent across different
        # runs, so we can't use it to identify the specific properties of a
        # Fit object. In contrast, hashlib's functions are consistent.

        return int(hashlib.sha256(str(hash_data).encode("utf-8")).hexdigest(), 16)


    @staticmethod
    def set_cache_folder(path):
        """
        """
        global _cache_enabled, _cache_path

        # First, create the folder if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save the path and set caching enabled
        _cache_enabled = True
        _cache_path = os.path.abspath(path)


    @staticmethod
    def set_cache_format(format):
        """
        """
        global _cache_format
        if format in SUPPORTED_SAVE_FILE_EXTENSIONS:
            _cache_format = format

        else:
            raise ValueError(f'Invalid save file format provided: {format}. Available formats are: {SUPPORTED_SAVE_FILE_EXTENSIONS}')


def _write_hdf5_file(filename, fit):
    """
    Save a fit object to a file, using the hdf5 file format.

    Parameters
    ----------
    filename : str or Path
        The path to the file that will be created.

    fit : Fit
        The fit object.
    """
    with h5py.File(filename, 'w') as f:
        # Create the main dataset
        dataset = f.create_dataset('data', data=fit.data_original)

        # h5 files can't save Python's NoneType (since they should
        # work with any language) so we have to do a little parsing.
        metadata = {}
        metadata["xmin"] = fit.xmin
        metadata["fixed_xmin"] = fit.fixed_xmin
        metadata["xmax"] = fit.xmax if fit.xmax else np.nan

        metadata["discrete"] = fit.discrete
        metadata["fit_method"] = fit.fit_method
        metadata["estimate_discrete"] = fit.estimate_discrete if fit.discrete else False
        metadata["discrete_normalization"] = fit.discrete_normalization
        metadata["sigma_threshold"] = fit.sigma_threshold if fit.sigma_threshold else np.nan
        metadata["xmin_distribution_name"] = fit.xmin_distribution_cls.name
        metadata["xmin_distance"] = fit.xmin_distance
        metadata["test_all_xmin"] = fit.test_all_xmin
        metadata["noise_flag"] = getattr(fit, 'noise_flag', False)

        # For the initial parameters and parameter ranges, we're going
        # to have to unpack each entry in those dictionaries (since we
        # can't store a proper dictionary).
        if fit.initial_parameters:
            for k, v in fit.initial_parameters.items():
                metadata[f"initial_{k}"] = v if v else np.nan

        if fit.parameter_ranges:
            for k, v in fit.parameter_ranges.items():
                metadata[f"range_{k}_min"] = v[0] if v[0] else np.nan
                metadata[f"range_{k}_max"] = v[1] if v[1] else np.nan

        # The parameter constraints are quite tricky as well, since we
        # can't store a Python function object in an h5 file. Instead, we
        # just copy the source code for the constraint. Note that this
        # does mean you can't use values defined outside of your constraint
        # function, for example:
        # 
        # some_value = 5
        # def constraint(dist):
        #    return dist.value < some_value
        #
        # This above constraint function cannot be saved properly because
        # some_value is defined outside of its scope.
        if fit.parameter_constraints:
            for constr in fit.parameter_constraints:
                function_source = inspect.getsource(constr["fun"])
                function_name = constr["fun"].__name__

                # This serialized function is the thing we will actually
                # use to reconstruct the function. It can also be done
                # by just executing the source code, though this will
                # lose access to any variable defined outside of the function.
                # The serialized code will include those.
                serialized_function = pickle.dumps(constr["fun"])

                # But the serialized function isn't easily readable, so we
                # want to save the source code too.
                function_data = np.array([function_source, serialized_function], dtype='S')

                # We create a new dataset for each constraint, with the
                # main data being the function source code
                constraint_dataset = f.create_dataset(f'constraint_{function_name}', data=function_data)

                constraint_dataset.attrs.update({'type': constr["type"],
                                                 'dists': constr.get("dists", ''),
                                                 'name': function_name})

        # Include the xmin fitting results if available.
        if hasattr(fit, 'xmin_fitting_results'):
            # We'll make a new dataset folder with entries for each of
            # the arrays

            # We also save the minimum index of these arrays in each
            # one. This might be a little redundant, but again the idea
            # of using h5 in the first place is that it is accessible
            # outside of this library and outside of even Python.
            distances = f.create_dataset('xmin_fitting_results/distances', data=fit.xmin_fitting_results["distances"])
            distances.attrs['min_index'] = fit.xmin_fitting_results["min_index"]

            xmins = f.create_dataset('xmin_fitting_results/xmins', data=fit.xmin_fitting_results["xmins"])
            xmins.attrs['min_index'] = fit.xmin_fitting_results["min_index"]

            valid_fits = f.create_dataset('xmin_fitting_results/valid_fits', data=fit.xmin_fitting_results["valid_fits"])
            valid_fits.attrs['min_index'] = fit.xmin_fitting_results["min_index"]

            # Now we have to create arrays for the parameters of the specific
            # xmin distribution we are using
            xmin_fitting_parameters = fit.xmin_distribution_cls.parameter_names
            for param in xmin_fitting_parameters:
                param_dataset = f.create_dataset(f'xmin_fitting_results/{param}', data=fit.xmin_fitting_results[param])
                param_dataset.attrs['min_index'] = fit.xmin_fitting_results["min_index"]

        # Finally, we include the version, since we want to warn if we
        # open a file made in another version.
        metadata["powerlaw_version"] = POWERLAW_VERSION if POWERLAW_VERSION else 'Unknown'

        dataset.attrs.update(metadata)


def _write_pickle_file(filename, fit):
    """
    Save a fit object to a file, using the Python pickle format.

    Parameters
    ----------
    filename : str or Path
        The path to the file that will be created.

    fit : Fit
        The fit object.
    """
    with open(filename, 'wb') as f:
        pickle.dump((fit, POWERLAW_VERSION), f)


def _parse_hdf5_file(filename, verbose=1):
    """
    Parse an hdf5 file created using the ``powerlaw.Fit.save()`` function. 

    Not intended to be called directly, but rather through
    ``powerlaw.Fit.load()``.

    Parameters
    ----------
    filename : str or Path
        The path to the file that will be loaded.

    Returns
    -------
    fit : Fit
        The loaded fit object, which should be functionally identical
        to the saved object.
    """
    with h5py.File(filename) as f:
        data = f["data"][:]
        # Most of the metadata here we will directly pass to the Fit
        # object, but we do have to parse a few things.
        metadata = dict(f["data"].attrs)

        # Parse the initial parameter values
        initial_parameters = {}
        for k, v in metadata.items():
            if "initial_" in k:
                variable_name = k.split('_')[1]
                initial_parameters[variable_name] = v if (not np.isnan(v)) else None

        if len(initial_parameters) == 0:
            initial_parameters = None

        # Parse the parameter ranges
        # These will be entries of the form "range_{var}_min" or "range_{var}_max"
        parameter_ranges = {}
        for k, v in metadata.items():
            if "range_" in k:
                if k.split('_')[-1] not in ["max", "min"]:
                    warnings.warn(f'Malformed h5 file formatting for parameter ranges: {k}:{v}')
                    continue

                variable_name = k.split('_')[1]
                range_index = int(k.split('_')[-1] == "max")

                current_range = parameter_ranges.get(variable_name, [None, None])
                current_range[range_index] = float(v) if (not np.isnan(v)) else None

                parameter_ranges[variable_name] = current_range

        if len(parameter_ranges) == 0:
            parameter_ranges = None

        # Parse the parameter constraints
        parameter_constraints = []
        for dataset in f.keys():
            if "constraint_" in dataset:
                function_source = f[dataset][:][0]
                serialized_function = f[dataset][:][1]

                # Execute the source code to load the function in
                try:
                    # Note that we have to "dedent" this function since it
                    # will maintain any indentation from exactly where
                    # it was written. This gets rid of extra indentation
                    # and puts the "def ..." line at zero indent.
                    #exec(textwrap.dedent(function_source.decode('utf-8')))
                    # It is better to use the serialization to reconstruct
                    # the function since it can save the values of variables
                    # defined outside of the function.
                    function = pickle.loads(serialized_function)

                except:
                    raise ValueError(f'Malformed constraint function {dataset} in file {filename}.')

                constraint_metadata = dict(f[dataset].attrs)

                function_type = constraint_metadata["type"]
                dists = constraint_metadata["dists"]

                constraint_dict = {"type": function_type,
                                   "fun": function}
                if len(dists) > 0:
                    constraint_dict["dists"] = list(dists)

                parameter_constraints.append(constraint_dict)


        if len(parameter_constraints) == 0:
            parameter_constraints = None

        # Parse xmin fitting results. These are stored in separate
        # datasets within the folder (within the file) 'xmin_fitting_results'.
        if not metadata["fixed_xmin"]:

            xmin_fitting_results = {}
            xmin_fitting_results["distances"] = f["xmin_fitting_results/distances"][:]
            xmin_fitting_results["valid_fits"] = f["xmin_fitting_results/valid_fits"][:]
            xmin_fitting_results["xmins"] = f["xmin_fitting_results/xmins"][:]

            xmin_fitting_results["min_index"] = f["xmin_fitting_results/xmins"].attrs["min_index"]

            # Now get the parameter arrays
            xmin_fitting_parameters = SUPPORTED_DISTRIBUTIONS[metadata["xmin_distribution_name"]].parameter_names
            for param in xmin_fitting_parameters:
                xmin_fitting_results[param] = f[f"xmin_fitting_results/{param}"][:]


        # There are various issues that might arise if you use a different
        # version of the package from the one that created the cached
        # file. For example, if there was a bug in some calculation
        # that was fixed in a newer version, you might not recalculate
        # the value to fix the issue. Most of the time you probably
        # shouldn't have issues, but we should give a warning just in
        # case.
        if POWERLAW_VERSION.lower() != "unknown" and metadata["powerlaw_version"].lower() != "unknown":
            if POWERLAW_VERSION != metadata["powerlaw_version"]:
                warnings.warn(f'Cached file {filename} was saved with a different version of powerlaw {metadata["powerlaw_version"]} than what you are currently using {POWERLAW_VERSION}! This may cause issues...')

        # Create the fit object
        fit = Fit(data=data,
                  discrete=bool(metadata["discrete"]),
                  xmin=metadata["xmin"],
                  xmax=metadata["xmax"] if (not np.isnan(metadata["xmax"])) else None,
                  fit_method=metadata["fit_method"],
                  estimate_discrete=bool(metadata["estimate_discrete"] if metadata["discrete"] else None),
                  discrete_normalization=metadata["discrete_normalization"],
                  sigma_threshold=metadata["sigma_threshold"] if (not np.isnan(metadata["sigma_threshold"])) else None,
                  initial_parameters=initial_parameters,
                  parameter_ranges=parameter_ranges,
                  parameter_constraints=parameter_constraints,
                  xmin_distance=metadata["xmin_distance"],
                  xmin_distribution=metadata["xmin_distribution_name"],
                  test_all_xmin=bool(metadata["test_all_xmin"]),
                  ignore_cache=True, # We have to ignore cacheing otherwise we'll get an infinite loop.
                  verbose=verbose)

        # Now we have to adjust the fact that maybe we did actually do
        # xmin fitting (but we just cached the result)
        if not metadata["fixed_xmin"]:
            fit.fixed_xmin = False
            fit.noise_flag = metadata["noise_flag"]

            fit.xmin_fitting_results = xmin_fitting_results
    
            # Set the Fit's xmin to the optimal xmin
            setattr(fit, metadata["xmin_distance"], xmin_fitting_results["distances"][xmin_fitting_results["min_index"]])

            # Update the fitting CDF given the new xmin, in case other objects, like
            # Distributions, want to use it for fitting (like if they do KS fitting)
            fit.fitting_cdf_bins, fit.fitting_cdf = fit.cdf()

    return fit


def _parse_pickle_file(filename):
    """
    Parse a pickle file created using the ``powerlaw.Fit.save()`` function. 

    Not intended to be called directly, but rather through
    ``powerlaw.Fit.load()``.

    Parameters
    ----------
    filename : str or Path
        The path to the file that will be loaded.

    Returns
    -------
    fit : Fit
        The loaded fit object, which should be functionally identical
        to the saved object.
    """
    with open(filename, 'rb') as f:
        # Note that we save the version with the fit object
        fit, version = pickle.load(f)

    # There are various issues that might arise if you use a different
    # version of the package from the one that created the cached
    # file. For example, if there was a bug in some calculation
    # that was fixed in a newer version, you might not recalculate
    # the value to fix the issue. Most of the time you probably
    # shouldn't have issues, but we should give a warning just in
    # case.
    if POWERLAW_VERSION.lower() != "unknown" and version.lower() != "unknown":
        if POWERLAW_VERSION != version:
            warnings.warn(f'Cached file {filename} was saved with a different version of powerlaw {version} than what you are currently using {POWERLAW_VERSION}! This may cause issues...')

    return fit
