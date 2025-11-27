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
    An abstract class for theoretical probability distributions.

    Can be created with particular parameter values, or fitted to a dataset.

    Fitting is by maximum likelihood estimation by default, though can also
    be done by minimizing distance metrics (eg. Kolmogorov-Smirnov) between
    the theoretical and actual distribution.

    Parameters
    ----------
    data : list or array, optional
        The data to which to fit the distribution. If provided, the fit will
        be created at initialization.

    xmin : int or float
        The data value beyond which the distribution is defined.

        Required for any distribution.

    xmax : int or float, optional
        The upper bound of the distribution.

        If not provided, the distribution will be assumed to be unbounded
        on the upper side.

    discrete : boolean, optional
        Whether the distribution is discrete (integers).

        Various approximations can be employed when there isn't an exact
        (or even approximate) expression for the PDF or CDF in the 
        discrete case. See ``discrete_normalization`` for more information.

        Some distributions have approximation expressions for the parameters
        and/or RNG in discrete cases as well; see ``estimate_discrete``.

    fit_method : {"likelihood", "ks"}, optional
        Method for fitting the distribution. ``"likelihood"`` is maximum likelihood
        estimation. ``"ks"`` is minimial distance estimation using the
        Kolmogorov-Smirnov test.

    parameters : array_like or dict, optional
        The parameters of the distribution. If data is given, these will
        be used as the initial parameters for fitting; otherwise, they
        will be taken as the final parameters for the distribution.

        Parameters can also be passed as keywords. Valid keywords for each
        distribution can be found in the ``parameter_names`` property of
        each distribution.

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

    discrete_normalization : {"round", "sum"}, optional
        Approximation method to use in calculating the PDF (especially the
        PDF normalization constant) for a discrete distribution in the case
        that there is no analytical expression available.

        ``"round"`` uses the probability mass in the region ``[x - 0.5, x + 0.5]`` for each
        data point ``x``.

        ``"sum"`` simply sums the PDF over the defined range to compute the
        normalization.

    parent_Fit : Fit object, optional
        A Fit object from which to use data, if it exists.

    verbose : {0, 1, 2}, bool, optional
        Whether to print debug and status information. ``0`` or ``False`` means
        print no information (including no warnings), ``1`` or ``True`` means print
        only warnings, and ``2`` means print warnings and status messages.

    kwargs :
        Parameter values for specific distributions can be passed as
        keyword arguments.
    """

    def __init__(self,
                 data=None,
                 xmin=None,
                 xmax=None,
                 discrete=False,
                 fit_method='likelihood',
                 parameters=None,
                 parameter_ranges=None,
                 parameter_constraints=None,
                 discrete_normalization='round',
                 parent_Fit=None,
                 verbose=1,
                 **kwargs):


        # When defining a subclass of this one, you should define this
        # list as a property of the class
        #parameter_names = []
        # You also will need to define this in the subclass with the
        # upper and lower bound for each parameter.
        #DEFAULT_PARAMETER_RANGES = {}
        # (but we don't define them here because we don't want to overwrite
        # the values created in the subclass)
        # And the name of the distribution
        #name = 'name'

        self.verbose = verbose

        self.fit_method = fit_method
        self.discrete = discrete
        self.discrete_normalization = discrete_normalization

        self.data = data

        # If our data is only integer values, but discrete is not true
        # or vice versa, give a warning
        if hasattr(self.data, '__iter__') and self.discrete and any(data != np.asarray(data, dtype=np.int64)):
            warnings.warn('discrete=True but data does not exclusively contain integer values. Casting to integer...')

        elif hasattr(self.data, '__iter__') and (not self.discrete) and all(data == np.asarray(data, dtype=np.int64)):
            warnings.warn('discrete=False but data exclusively contains integer values. Consider using discrete=True.')

        # We need an xmin, so we just set it to the minimum value of the
        # data if we don't get a specific value
        if not xmin:
            if hasattr(self.data, '__iter__'):
                self.xmin = np.min(data)

            # If we don't have data, then we need to raise an error.
            else:
                raise ValueError('No xmin provided, and no data to infer value from. Provide an explicit value for xmin!')

        else:
            self.xmin = xmin
        self.xmax = xmax

        # If we don't have a parent fit, we still have to make sure that
        # this variable gets assigned, otherwise we'll have logic issues
        # later on.
        self.parent_Fit = parent_Fit

        if self.parent_Fit and not hasattr(self.data, '__iter__'):
            self.data = self.parent_Fit.data

        # Crop the data to the domain, but make a copy of the original
        # first. This means we don't need to trim_to_range whenever
        # we access the data in the future.
        if hasattr(self.data, '__iter__'):
            self.original_data = np.copy(data)
            self.data = trim_to_range(self.data, xmin=xmin, xmax=xmax)

        # Setup the initial parameters and things
        # We have to pass the kwargs here because parameters might be
        # directly passed as keywords.
        self.initialize_parameters(parameters, **kwargs)

        # Setup the parameter ranges
        # This sets the variable `self.parameter_ranges`
        self.initialize_parameter_ranges(parameter_ranges)

        # Setup parameter contraints
        self.initialize_parameter_constraints(parameter_constraints)

        # This will be set in fit() to True if the fitting failed.
        self.noise_flag = False

        # Fit if we have data
        if hasattr(self.data, '__iter__'):

            self.n = len(self.data)
            self.fit(data)

        self.debug_parameter_names()

    
    def debug_parameter_names(self):
        """
        This is solely a debug function that sets up variables in the
        old format (from v1.5, eg. ``self.parameter1``).

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


    def initialize_parameters(self, initial_parameters=None, **kwargs):
        """
        Set up the parameters for the distribution.

        If parameters are passed, they will try to be parsed, otherwise
        initial guesses for parameters specific to each distribution will
        be used.

        Note that there is also a function ``set_parameters()`` in this class.
        The primary difference between the two is that this function allows
        the possibility of generating initial guesses of parameters
        (using the ``generate_initial_parameters()`` function), whereas
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
            ``self.parameter_names``, and then the values in that list will
            be assumed as the initial values for the parameters in the
            same order as the former list.

            If not provided, the values will be initialized using
            ``self.generate_initial_parameters()`` which will try its best
            to give a reasonable start based on the data.

        kwargs : 
            Parameters can be passed as direct keyword arguments as well.
        """
        # Since the user may only specify a few parameters (not required to
        # set all of them, we should start with the default values).
        # This is only possible when we have data though
        data = self.data
        if hasattr(data, '__iter__'):
            # If we aren't given any initial parameters, try to generate
            # them from the data.

            initial_parameters_dict = self.generate_initial_parameters(data)

        else:
            initial_parameters_dict = dict(zip(self.parameter_names, [None]*len(self.parameter_names)))

        # If we were given a dictionary of initial parameters, we use
        # those values as is.
        if type(initial_parameters) == dict:
            # Only choose the parameters that are actually for this class
            # (since we may be given parameters from a fit object that
            # contains information for other distributions).
            for k,v in initial_parameters.items():
                if k in self.parameter_names:
                    initial_parameters_dict[k] = v

        elif hasattr(initial_parameters, '__iter__') and len(initial_parameters) == len(self.parameter_names):
            # If we are given a list of initial parameters, we assume
            # the order of them is the same as the order of self.parameter_names.
            initial_parameters_dict.update(dict(zip(self.parameter_names, initial_parameters)))

        elif initial_parameters is None:
            # If initial_parameters is None, it's possible that the user
            # passed the parameter values through direct keywords.
            for kw in kwargs.keys():
                if kw in self.parameter_names:
                    initial_parameters_dict[kw] = kwargs[kw]

            # It's also possible they just want to use the generated values
            # from generate_initial_parameters, in which case we do nothing
            # here.

        else:
            # Otherwise, something must be wrong with the initial parameters
            # provided, so we raise an error.
            raise Exception(f'Invalid value provided for initial parameters: {initial_parameters}.')

        # Let's make sure that we got some values, since it's possible we
        # could get to this point by passing no parameters and having
        # no data.
        if any([value is None for value in initial_parameters_dict.values()]):
            raise Exception('Not enough information provided to assign parameters! Make sure to specific parameter \
                            values or pass data to generate them.')

        # Actually set the values
        for p in self.parameter_names:
            setattr(self, p, initial_parameters_dict[p])
        
        # We also replace initial_parameters with the proper dictionary,
        # since that is probably more useful
        self.initial_parameters = initial_parameters_dict


    def set_parameters(self, params):
        """
        Set the parameters for the distribution.

        Intended to be called during fitting, as opposed to
        ``initialize_parameters()`` which is designed to be called during
        construction. See documentation for this other function for more
        information.

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
            ``self.parameter_names``, and then the values in that list will
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
        A dictionary in which each key corresponds to a parameter
        name and the value corresponds to the initial value of that
        parameters.
        """
        return dict(zip(self.parameter_names, [getattr(self, p) for p in self.parameter_names]))


    def initialize_parameter_ranges(self, ranges=None):
        """
        Set up the ranges for parameters for the distribution.

        If not provided, the default range for each parameter will be
        used; these are specific to each individual distribution. For
        more information, see the variable ``DEFAULT_PARAMETER_RANGES`` in
        children of this class.

        Parameters
        ----------
        ranges : dict or array-like, optional
            A dictionary in which each key corresponds to a parameter
            name and the value corresponds to the lower and upper bound
            for that parameter (in the format of a length 2 tuple/list/array).
            Ranges can be provided for only some parameters.

            Can also be given as a list that is exactly the length of
            ``self.parameter_names``, and then the values in that list will
            be assumed as the lower and upper bound for the parameter
            with that same index in ``self.parameter_names``. As such,
            ranges must be given for all parameters if given as a list.

            If not provided, the values will be initialized using
            ``self.DEFAULT_PARAMETER_RANGES``.
        """
        # We start with the default range, since it's possible that the
        # user has passed a range for only one or two parameters instead
        # of all of them. In that case, the parameters that haven't been
        # specified should follow their default range. Also we make a copy
        # so we don't change the original
        ranges_dict = dict(self.DEFAULT_PARAMETER_RANGES)

        # If we were given a dictionary of ranges, we use
        # those values as is.
        if type(ranges) == dict:
            # Only choose the parameter ranges that are actually for this class
            # (since we may be given parameters from a fit object that
            # contains information for other distributions).
            for k,v in ranges.items():
                if k in self.parameter_names:
                    ranges_dict[k] = v

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
            are the initial values of each parameter.
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


    def initialize_parameter_constraints(self, parameter_constraints):
        """
        Parse and save constraint functions for parameters to be used
        during the fitting process.

        The constraints are used in ``scipy.optimize.minimize()`` and so
        the end result follows the requirements for dictionary-type
        constraints described here, though with some slight modifications:

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Parameters
        ----------
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
        """
        # The final constraint object we want is a list of dictionaries that can
        # be passed to scipy.optimize.minimize.

        # That being said, we have a few extra pieces of information we
        # want to pass, so even if we are passed a dictionary already, we
        # should parse all of the information, then create a new dictionary.

        # From each (if multiple) constraint function, we need the type,
        # ('eq' or 'ineq'), the function itself, and whether that constraint
        # applies to this distribution.
        functions_list = []
        type_list = []

        constraint_list = []

        # First, make a temporary list of the dictionaries if we don't already
        # have one.
        if hasattr(parameter_constraints, '__iter__') and type(parameter_constraints) is not dict and len(parameter_constraints) > 0:
            constraint_list = parameter_constraints

        elif type(parameter_constraints) is dict:
            constraint_list = [parameter_constraints]
       
        # Make sure each entry is a dictionary (this will pass if the
        # length of constraint_list is zero).
        assert all([type(con) is dict for con in constraint_list]), f'List of constraints passed to {self.name} but not all are dictionaries.'

        # Now we parse the information from each dictionary
        for con in constraint_list:
            # This whole thing is wrapped in a try statement so we
            # can return a clearer error if the dictionary is formatted
            # weirdly.
            try:
                # If the dictionary has an entry 'dists', this should be
                # a list of the distributions to which this constraint
                # applies. So we only save the constraint if this entry
                # doesn't exist (implying it applies to all distributions)
                # or if the name of this dist is in there.
                if not 'dists' in con:
                    functions_list.append(con["fun"])
                    type_list.append(con["type"])

                elif 'dists' in con and hasattr(con["dists"], '__iter__') and self.name in con["dists"]:
                    functions_list.append(con["fun"])
                    type_list.append(con["type"])

                # Otherwise, we ignore this constraint
                else:
                    pass

            except:
                raise Exception(f'Malformed constraint dictionary passed to {self.name}: received {con}')

        # The final list of dictionaries we are building
        constraint_dict_list = []

        # Now that we have the constraints set up, we wrap them in an
        # outer function which allows us to pass the fit object as the
        # argument instead of the list of parameter values.
        for i in range(len(functions_list)):

            # Globals obviously aren't good practice, but this is the
            # only way I can imagine that we gain access to the whole
            # Distribution object within the constraint function in
            # scipy.optimize.minimize.
            function = functions_list[i]
            global class_instance
            class_instance = self

            def wrapped_function(params):
                # We aren't actually going to use these params
                # but will load our global class instance and pass that.
                global class_instance
                return function(class_instance)

            constraint_dict_list.append({"type": type_list[i],
                                         "fun": wrapped_function})

        # Save the constraints
        self.parameter_constraints = constraint_dict_list


    def fit(self, data=None):
        """
        Numerically fits the parameters of the distribution to the data.

        Fitting is performed by minimizing either the loglikelihood or
        KS distance (depending on the value of ``Distribution.fit_method``).

        Bounds defined in ``self.parameter_ranges`` are used explicitly during
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

        # If we still don't have data, raise an error
        if not hasattr(data, '__iter__'):
            raise ValueError(f'No data to fit distribution ({self.name}) to!')

        # self.data is already trimmed, but if we are passed new data
        # we need to trim it.
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

        # In choosing the minimization method, unfortunately there isn't
        # one choice that works for all cases. The old method (see below)
        # uses Nelder-Mead.

        # Powell and Nelder-Mead 
        # together cover most options though is what I've found.
        # In particular, Powell can traverse the inflection point at
        # alpha = 1 only going down (ie works for any alpha <= 1) whereas
        # Nelder-Mead can only traverse it going up (ie. works for any
        # alpha > 1).

        # The only options I've found that have success both ways is
        # COBYLA and COBYQA. COBYQA struggles around alpha = 1, but
        # COBYLA seems relatively fine, but struggles with lognormal fitting.
        # I also find that it actually does a better job of fitting
        # exponentials, but this means that the tests fail because it
        # does better than it used to...

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

        # After testing, I think sticking with Nelder-Mead for now is fine,
        # unless we have a constraint, then we use COBYLA (since Nelder-Mead
        # can't handle constraints). But this should definitely be reviewed
        # later on.
        #methods = ["Powell", "Nelder-Mead"]

        if hasattr(self.parameter_constraints, '__iter__') and len(self.parameter_constraints) > 0:
            #methods = ["COBYLA"]
            methods = ["SLSQP"]
        else:
            methods = ['Nelder-Mead']

        # Only switch back and forth if we have more than one method.
        switches = 2 if len(methods) > 1 else 1

        # Take the initial parameters
        parameters = list(self.parameters.values())

        for i in range(switches):
            for m in methods:
                result = scipy.optimize.minimize(fit_function,
                                                 x0=parameters,
                                                 bounds=bounds,
                                                 method=m,
                                                 constraints=self.parameter_constraints,
                                                 tol=1e-4)
                parameters = result.x

        # In case you'd like to compare, this uses the exact same optimization
        # as in powerlaw v1.5 and before (Nelder-Mead). The results end up pretty much
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
        # If you switch to the old optimization method, make sure to
        # comment out the second term in this expression.
        self.noise_flag = (not self.in_range()) or (not result.success)
  
        if self.noise_flag and self.verbose:
            warnings.warn(f"No valid fits found for distribution {self.name}.")

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
                warnings.warn(f'Fitted parameters are very close to the edge of parameter ranges for distribution {self.name}; consider changing these ranges.')


    def KS(self, data=None):
        """
        Return the Kolmogorov-Smirnov distance D.

        Included for backwards compatability; this method has been
        renamed ``compute_distance_metrics()`` because it computes several
        distance metrics (including KS).
        """
        compute_distance_metrics(data)
        return self.D

    
    def compute_distance_metrics(self, data=None):
        r"""
        Compute various distance metrics between the fit distribution and
        the actual distribution of data.

        The following distance metrics will be computed, with :math:`C_d`
        and :math:`C_t` as the cumulative distribution function for the
        data and for the theoretical fit, respectively.

        Kolmogorov-Smirnov distance:

        .. math::
            D = max( max(C_d - C_t), -min(C_d - C_t) )
         
        Kuiper distance:

        .. math::
            V = max(C_d - C_t) - min(C_d - C_t)

        Anderson-Darling distance:

        .. math::
            A^2 = \sum( (C_d - C_t)^2 / (C_t (1 - C_t)))

        Kappa:

        .. math::
            K = 1 + mean(C_d - C_t)

        The names of these distance metrics will be stored as ``D``, ``V``, 
        ``Asquare``, and ``Kappa`` respectively.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data passed on creation
            of the Distribution object.
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        # If we still don't have data, raise an error
        if not hasattr(data, '__iter__'):
            raise ValueError(f'No data to compute distance metrics for in distribution {self.name}!')

        # self.data is already trimmed, but if we are passed new data
        # we need to trim it.
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)

        # If we don't have enough data, return a bunch of nan values
        if len(data) < 2:
            warnings.warn("Not enough data to compute distance metrics like Kolmogorov-Smirnov distance, returning nan.")

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


    def ccdf(self, data=None):
        """
        The complementary cumulative distribution function (CCDF) of the
        theoretical distribution. Calculated for the values given in data
        between xmin and xmax, if present.

        Parameters
        ----------
        data : list or array, optional
            The data for which to compute the CCDF. If not provided, the data
            passed on creation (if available) will be used.

        Returns
        -------
        probabilities : numpy.ndarray
            The portion of the data that is greater than or equal to X.
        """
        return 1 - self.cdf(data=data)


    def cdf(self, data=None):
        """
        The cumulative distribution function (CDF) of the theoretical
        distribution. Calculated for the values given in data within xmin and
        xmax, if present.

        Parameters
        ----------
        data : array_like, optional
            The data for which to compute the CDF. If not provided, the data
            passed on creation (if available) will be used.

        Returns
        -------
        probabilities : numpy.ndarray
            The portion of the data that is less than or equal to X.
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        # self.data is already trimmed, but if we are passed new data
        # we need to trim it.
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)

        n = len(data)

        if n == 0:
            raise Exception('No data points in defined range of the distribution.')

        # If we aren't in range, we just return a bunch of (nearly) zeros
        if not self.in_range():
            return np.tile(10**sys.float_info.min_10_exp, n)

        if self._cdf_xmin == 1:
            # If cdf_xmin is 1, it means we don't have the numerical accuracy to
            # calculate this tail. So we make everything 1, indicating
            # we're at the end of the tail. Such an xmin should be thrown
            # out by the KS test.
            CDF = np.ones(n)
            return CDF

        CDF = self._cdf_base_function(data) - self._cdf_xmin

        norm = 1 - self._cdf_xmin
        # If we have an xmax, our normalization is slightly different
        if self.xmax:
            norm = norm - (1 - self._cdf_base_function(self.xmax))

        CDF = CDF/norm

        # If we have any nan values in the cdf, it is indicative of a
        # numerical error, so we should warn.
        # np.min will always give nan if there are any nans
        possible_numerical_error = np.isnan(np.min(CDF))

        if possible_numerical_error and self.verbose:
            warnings.warn("Likely underflow or overflow error: the optimal fit for this distribution gives values that are so extreme that we lack the numerical precision to calculate them.")

        return CDF


    @property
    def _cdf_xmin(self):
        """
        The CDF evaluated at the point ``xmin``; also the minimum value
        of the CDF.
        """
        return self._cdf_base_function(self.xmin)


    def pdf(self, data=None):
        """
        The probability density function (normalized histogram) of the
        theoretical distribution for the values in data within ``xmin`` and
        ``xmax``, if present.

        Parameters
        ----------
        data : array_like, optional
            The data for which to compute the PDF. If not provided, the data
            passed on creation (if available) will be used.

        Returns
        -------
        probabilities : numpy.ndarray
            The portion of the data that is contained at each bin (data point).
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        # self.data is already trimmed, but if we are passed new data
        # we need to trim it.
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)

        n = len(data)

        if n == 0:
            raise Exception('No data points in defined range of the distribution.')

        # If we aren't in range, we just return a bunch of (nearly) zeros
        if not self.in_range():
            return np.tile(10**sys.float_info.min_10_exp, n)

        # If we have a continuous distribution, we can easily apply
        # our base function and the normalization factor to get the
        # pdf evaluated at each data point.
        if not self.discrete:
            f = self._pdf_base_function(data)
            C = self._pdf_continuous_normalizer

            likelihoods = f*C

        # For discrete cases, it's a little more tricky since we will
        # have to approximate the normalization factor.
        else:
            # If we have an explicit expression for the discrete
            # normalization, we should of course use that.
            if self._pdf_discrete_normalizer:
                f = self._pdf_base_function(data)
                C = self._pdf_discrete_normalizer

                likelihoods = f*C

            elif self.discrete_normalization == 'round':
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

                likelihoods = self.cdf(upper_data) - self.cdf(lower_data)

                self.xmin += 0.5
                if self.xmax:
                    self.xmax -= 0.5

            elif self.discrete_normalization == 'sum':
                # Otherwise, we just normalize numerically by summing the
                # PDF

                # There used to be the option to pass a specific upper_limit
                # value in the discrete_normalization keyword, but I don't
                # think that is actually useful. Passed values above xmax
                # wouldn't be useful since we don't have any data there, 
                # and values below xmax would be confusing since there is
                # already a well-defined upper limit here: xmax.
                if self.xmax:
                    upper_limit = self.xmax
                else:
                    upper_limit = np.max(self.data)

                # Compute the pdf for all possible values, ie. assuming
                # we have perfectly sampled our distribution.
                X = np.arange(self.xmin, upper_limit+1)
                PDF = self._pdf_base_function(X)

                # Then normalize using the 'perfect' distribution, but
                # only take PDF values where we actually have data.
                PDF = (PDF / np.sum(PDF)).astype(float)
                likelihoods = PDF[(data - self.xmin).astype(int)]

        # Set any zeros to very small values
        likelihoods[likelihoods == 0] = 10**sys.float_info.min_10_exp

        # We cast to float64 since the previous type might have been
        # float128 (longdouble) and that causes issues with special
        # functions like erfc
        return likelihoods.astype(np.float64)


    @property
    def _pdf_continuous_normalizer(self):
        """
        The (inverse of the) normalization constant for the PDF
        assuming a continuous distribution. So you can get the normalized
        PDF evaluated for a particular ``x`` as:

            _pdf_continuous_normalizer * _pdf_base_function(x)

        The implementation below simply uses the cumulative distribution
        function to compute this, though if an explicit expression can
        be found by integrating the PDF, this function should be overwritten
        in the child class.
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
        The (inverse of the) normalization constant for the PDF
        assuming a discrete distribution.

        It is not currently implemented since the value will depend on your
        specific distribution, so it should be implemented in child classes.
        """
        return False


    def in_range(self):
        """
        Whether the current parameters (``self.parameters``) of the
        distribution are within the range of valid parameters (defined by
        ``self.parameter_ranges``) and satisfy any constraints provided
        (defined by ``self.parameter_constraints``)

        Returns
        -------
        result : bool
            True if all parameters are within the specified ranges and all
            constraints are satisfied.
        """
        # Final result of whether all of the parameters are in range
        result = True

        # Check if parameters are in range
        for p in self.parameter_names:
            if self.parameter_ranges[p][0] is not None:
                result *= getattr(self, p) > self.parameter_ranges[p][0]
            if self.parameter_ranges[p][1] is not None:
                result *= getattr(self, p) < self.parameter_ranges[p][1]

        # Check if constraints are satisfied.
        for con in self.parameter_constraints:
            # For equality constraints, we make sure the value is zero
            if con["type"] == 'eq':
                result *= con["fun"](list(self.parameters.values())) == 0

            # For inequality, we make sure the value is non negative
            else:
                result *= con["fun"](list(self.parameters.values())) > 0

        return bool(result)


    def likelihoods(self, data=None):
        """
        The likelihoods of the observed data from the theoretical distribution.
        Another name for the probabilities or probability density function.

        Parameters
        ----------
        data : array_like, optional
            The data for which to compute the PDF. If not provided, the data
            passed on creation (if available) will be used.

        Returns
        -------
        likelihoods : numpy.ndarray
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
            The data for which to compute the PDF. If not provided, the data
            passed on creation (if available) will be used.

        Returns
        -------
        likelihoods : numpy.ndarray
            Logarithm (base e) of likelihood of each observed data.
        """
        # No need to trim to range or check if data is None because pdf()
        # does that, which is called by self.likelihoods().
        return np.log(self.likelihoods(data))


    def plot_ccdf(self, data=None, ax=None, **kwargs):
        """
        Plot the complementary cumulative distribution function (CCDF) of the
        theoretical distribution for the values given in data within ``xmin``
        and ``xmax``, if present.

        Plots to a new figure or to axis ``ax`` if provided.

        Parameters
        ----------
        data : array_like, optional
            The data for which to compute the PDF. If not provided, the data
            passed on creation (if available) will be used.

        ax : matplotlib axis, optional
            The axis on which to plot. If None, a new figure is created.

        kwargs
            Other keyword arguments are passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        ax : matplotlib axis
            The axis on which the plot was made.
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        # If we have data, we use that for the bins
        if hasattr(data, '__iter__'):
            bins = np.unique(trim_to_range(data, xmin=self.xmin, xmax=self.xmax))

        # Otherwise, we just generate bins from xmin to xmax if xmax exists,
        # or just xmin to xmin*1000. This 3 decades is arbitrary, but we
        # need to make some choice.
        else:
            if self.xmax:
                upper_bound = self.xmax

            else:
                upper_bound = self.xmin * 1e3

            # 1000 bins is arbitrary.
            bins = np.logspace(np.log10(self.xmin), np.log10(upper_bound), 1000)

        CCDF = self.ccdf(bins)

        if not ax:
            import matplotlib.pyplot as plt
            plt.plot(bins, CCDF, **kwargs)
            ax = plt.gca()

        else:
            ax.plot(bins, CCDF, **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")

        return ax


    def plot_cdf(self, data=None, ax=None, **kwargs):
        """
        Plot the cumulative distribution function (CDF) of the
        theoretical distribution for the values given in data within ``xmin``
        and ``xmax``, if present.

        Plots to a new figure or to axis ``ax`` if provided.

        Parameters
        ----------
        data : array_like, optional
            The data for which to compute the PDF. If not provided, the data
            passed on creation (if available) will be used.

        ax : matplotlib axis, optional
            The axis on which to plot. If None, a new figure is created.

        kwargs
            Other keyword arguments are passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        ax : matplotlib axis
            The axis on which the plot was made.
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        # If we have data, we use that for the bins
        if hasattr(data, '__iter__'):
            bins = np.unique(trim_to_range(data, xmin=self.xmin, xmax=self.xmax))

        # Otherwise, we just generate bins from xmin to xmax if xmax exists,
        # or just xmin to xmin*1000. This 3 decades is arbitrary, but we
        # need to make some choice.
        else:
            if self.xmax:
                upper_bound = self.xmax

            else:
                upper_bound = self.xmin * 1e3

            # 1000 bins is arbitrary.
            bins = np.logspace(np.log10(self.xmin), np.log10(upper_bound), 1000)

        CDF = self.cdf(bins)

        if not ax:
            import matplotlib.pyplot as plt
            plt.plot(bins, CDF, **kwargs)
            ax = plt.gca()

        else:
            ax.plot(bins, CDF, **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")

        return ax


    def plot_pdf(self, data=None, ax=None, **kwargs):
        """
        Plot the probability density function (PDF) of the
        theoretical distribution for the values given in data within ``xmin``
        and ``xmax``, if present.

        Plots to a new figure or to axis ``ax`` if provided.

        Parameters
        ----------
        data : array_like, optional
            The data for which to compute the PDF. If not provided, the data
            passed on creation (if available) will be used.

        ax : matplotlib axis, optional
            The axis on which to plot. If None, a new figure is created.

        kwargs
            Other keyword arguments are passed to `matplotlib.pyplot.plot()`.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        # If we have data, we use that for the bins
        if hasattr(data, '__iter__'):
            bins = np.unique(trim_to_range(data, xmin=self.xmin, xmax=self.xmax))

        # Otherwise, we just generate bins from xmin to xmax if xmax exists,
        # or just xmin to xmin*1000. This 3 decades is arbitrary, but we
        # need to make some choice.
        else:
            if self.xmax:
                upper_bound = self.xmax

            else:
                upper_bound = self.xmin * 1e3

            # 1000 bins is arbitrary.
            bins = np.logspace(np.log10(self.xmin), np.log10(upper_bound), 1000)

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


    def generate_random(self, size=1, estimate_discrete=None):
        """
        Generate random numbers from the theoretical probability distribution.

        This will follow the theoretical distribution, including upper
        and lower limits defined by ``xmin`` or ``xmax``. For example, if
        this function is called from a distribution with a finite value of
        ``xmax``, the generated values will be less than that value. If
        no value is given for ``xmax``, random values will have no upper
        limit.

        For discrete distributions without an approximation method, we
        use numerical inverse transform sampling.

        Parameters
        ----------
        size : tuple or int, optional
            The number of random numbers to generate.

            If a tuple, will be taken as the shape of the array to generate
            where each value is randomly generated according to the theoretical
            distribution.

        estimate_discrete : bool, optional
            For discrete distributions, whether to use a faster approximation of
            the random number generator.

            If ``None``, attempts to inherit the estimate_discrete behavior used
            for fitting from the ``Distribution`` object or the parent ``Fit``
            object, if present. Approximations only exist for some
            distributions (namely the power law). If an approximation does
            not exist, an ``estimate_discrete=True`` setting will not be inherited.

        Returns
        -------
        r : array
            Random numbers drawn from the distribution with shape equal
            to ``size``.
        """
        
        # For generating random numbers from an arbitrary distribution, we
        # use inverse transform sampling, which involves finding the inverse
        # of the cumulative distribution function, and then
        # evaluating that function for random uniform values in the domain
        # of the CDF, ie. [0, 1].

        # For continuous random numbers we usually don't need to do any
        # approximations, and can just transform according to the specific
        # distribution.
        if not self.discrete:
            # Note that assuming the full range [0, 1] will give unbounded
            # random numbers on the upper side; the distribution functions
            # (_generate_random_continuous) are derived assuming an xmin
            # value, but no xmax value.

            # So for the lower bound we can safely use zero when we don't have
            # an xmax value.
            lower_bound = 0
            upper_bound = 1

            if self.xmax:
                # When we have an xmax value, we need to change the upper
                # bound. This is because the whole cumulative distribution 
                # shifts when we have an xmax.

                # You might be tempted to use the self.cdf() function for that,
                # but this function already accounts for xmin and xmax, so it will
                # just give you 0 and 1, which we don't want. What we need is the
                # unadjusted cdf value, since the inverse transform sampling
                # function is derived without being adjusted for xmax.

                #upper_bound = self._cdf_base_function(self.xmax)

                # TODO: There is an issue with lognormal and exponential random
                # number generation, possibly because that inverse cdf is derived
                # differently than the others, but I don't understand why. For
                # lognormal or exponential generation to have proper bounds, you
                # need to somehow change the upper bound to some other value, but
                # I have no idea what that value is; it doesn't seem to be any
                # of the obvious combinations of self._cdf_base_function(self.xmin)
                # and self._cdf_base_function(self.xmax). As such, we just perform
                # a bisect search to find this value for ALL distributions;
                # for everything except exponential and lognormal, the value
                # of this should be identical to self._cdf_base_function(self.xmax).

                # Since this bisect search only needs to perform once per
                # random generation, this doesn't increase computation
                # that much, so long as you aren't generate one number at
                # a time. But of course we should still try to fix this issue.

                # We also have to make sure that our xmax isn't too large;
                # if we have defined a huge xmax when our distribution
                # goes to (nearly) zero much before it reaches this value,
                # we can't actually perform this search (because of
                # numerical precision). In that case, we can just use
                # 1 as an upper bound and call it a day :)

                # 1e-10 is arbitrary
                if 1 - self._cdf_base_function(self.xmax) <= 1e-10:
                    upper_bound = 1

                else:
                    upper_bound = bisect_map(mn=0, mx=1-1e-10,
                                             function=self._generate_random_continuous,
                                             target=self.xmax, tol=1e-8)

                    # Minus some epsilon since the bisect search isn't perfect
                    upper_bound -= 1e-6

            uniform_r = np.random.uniform(lower_bound, upper_bound, size=size)
            r = self._generate_random_continuous(uniform_r)

        # For discrete distributions, we usually have to make some
        # approximation.
        else:
            # Make sure that this distribution supports approximating the
            # continuous distribution with some discrete scheme.
            if estimate_discrete and not hasattr(self, '_generate_random_discrete_estimate'):
                raise AttributeError("This distribution does not have an estimation of the discrete form for generating simulated data. Try the exact form with estimate_discrete=False.")

            # If no value for estimate discrete is given, we should decide
            # based on whether the distribution is first able to do this
            # at all, then whether the class has already been passed a
            # value on creation.
            if estimate_discrete is None:
                
                # We can't estimate discrete is there isn't a function
                # for it.
                if not hasattr(self, '_generate_random_discrete_estimate'):
                    estimate_discrete = False

                # Check the value of self.estimate_discrete.
                elif hasattr(self, 'estimate_discrete'):
                    estimate_discrete = self.estimate_discrete

                # Check the value of estimate_discrete for the parent object.
                elif self.parent_Fit:
                    estimate_discrete = self.parent_Fit.estimate_discrete

                # If none of those worked, don't estimate.
                else:
                    estimate_discrete = False


            # Use the approximation method if it's available and
            # desired.
            if estimate_discrete:
                # Note that if we use the approximate discrete method, the
                # upper bound is different from a continuous one, since we
                # are using a different function than _generate_random_continuous.
                # So we first do a search to find the maximum value, ie.
                # the r value such that _generate_random_discrete_estimate(r) = xmax
                if self.xmax:
                    # For the upper limit mx here, we can't use exactly 1
                    # since that would lead to infinity for most distributions.
                    upper_bound = bisect_map(mn=0, mx=1-1e-15,
                                             function=self._generate_random_discrete_estimate,
                                             target=self.xmax - 1, # -1 to make sure we always generate values under xmax
                                             tol=1e-8)

                else:
                    upper_bound = 1

                # This function takes xmin into account, so we can just
                # use 0 as the lower bound.
                uniform_r = np.random.uniform(0, upper_bound, size=size)

                r = np.array(self._generate_random_discrete_estimate(uniform_r), dtype=np.int64)

            else:
                # For each of the uniform values (r), we do the
                # inverse search problem to find the specific value of x
                # where the ccdf is equal to that value of r. The x value
                # is then the random value we return.

                # This does the search on the function ccdf which will
                # automatically account for xmin and xmax, so we can just
                # use plain 0 and 1 for our bounds.
                uniform_r = np.random.uniform(0, 1, size=size)
                r = np.array([self._double_search_discrete(R) for R in uniform_r.flatten()], dtype=np.int64)

                # Now reshape
                r = r.reshape(size)

        return r


    def _double_search_discrete(self, r):
        """
        Perform the inverse search problem of locating the x value
        such that ccdf(x) = 1 - r.

        Parameters
        ----------
        r : float in [0, 1)
            A uniform random variable, representing the CDF value for which
            we want to find the corresponding x value.

        Returns
        -------
        x : float
            The sampled value from the theoretical distribution.
        """
        assert r >= 0 and r <= 1, f'Invalid r value provided to search for: {r}'

        # Find a range [x1, x2] that contains our random probability r
        x2 = int(self.xmin)
        while self.ccdf(data=[x2]) >= (1 - r):
            x1 = x2
            x2 = 2*x1

            # Make sure to clip the bound by xmax
            if self.xmax and x2 >= self.xmax:
                x2 = min(x2, self.xmax)
                # And end, since we can't go higher anymore
                break

        # Use bisect search within that range to find the integer that gives
        # a cdf value closest to the desired r (or rather, ccdf closest to 1 - r).
        func = lambda x: self.ccdf(data=[x])[0]
        # We use a tolerance of 1 since we care about the closest integer
        # value.
        x = bisect_map(mn=x1, mx=x2, function=func, target=1-r, tol=1)

        if x is None:
            print(x1, x2, 1-r)

        return int(np.around(x))


class Power_Law(Distribution):
    r"""
    A power law distribution, :math:`p(x) \sim x^{-\alpha}`.

    The exponent :math:`\alpha` should be positive, and is typically in the
    range :math:`(1, 3]`. For a normalizable distribution on an infinite or
    semi-infinite domain (ie. no :math:`x_{max}`), we should have alpha greater
    than 1.

    For continuous power laws with :math:`\alpha \in (1, 3]`, there is an
    exact expression for the maximum likelihood estimation (MLE) to a set of data [1].
    As such, numerical fitting is only performed when this expression isn't
    available, ie. when:

    1. The :math:`\alpha` value from this expression is less than 1 or
    greater than 3, indicating that the true :math:`\alpha` may be outside
    this range, or,
    2. There is a finite value for :math:`x_{max}`.

    For discrete power laws, there is an approximate expression for the
    :math:`\alpha` value from the MLE that is used under the following
    conditions:

    1. There is no value for :math:`x_{max}`,
    2. :math:`x_{min}` is greater than or equal to 10,
    3. The true alpha value of the distribution is in the range :math:`(1, 3]`, and
    4. The value from this expression falls in the range :math:`(1, 3]`.

    You can disable the use of this approximate expression with
    ``estimate_discrete=False``. These same considerations apply to the
    approximation discrete random number generator, which is also controlled
    using the same keyword.

    If you would like to perform numerical fitting after initializing
    the values with these above expressions --- in case you are worried
    they aren't good approximations for some particular case --- you can
    force this with ``force_numerical_fit=True``.

    Accepts all kwargs from ``Distribution`` super class.
    """
    name = 'power_law'

    parameter_names = ['alpha']

    DEFAULT_PARAMETER_RANGES = {'alpha': [0, 3]}


    def __init__(self, estimate_discrete=None, force_numerical_fit=False, **kwargs):
        """
        Parameters
        ----------
        estimate_discrete : bool, optional
            Whether to estimate alpha for discrete distributions using the
            approximate method described in [1].

            By default, this will be enabled when the error is of order 1%
            or less: according to the original paper, this is when there is
            no ``xmax`` and ``xmin <= 10``. Otherwise it will be disabled.

            Set this value to True or False to force the approximation to
            be used or not.

        force_numerical_fit : bool, optional
            Whether to force the fitting process to include numerical fitting
            even if an analytic expression is available. In this case, the
            result of the analytic expression will be used as the initial
            guess for the numerical fitting.

            See the documentation on this class for more information on
            when analytical or numerical fitting is used.

        """
        # This value will be parsed (and auto assigned if None) in
        # generate_initial_parameters.
        self.estimate_discrete = estimate_discrete

        self.force_numerical_fit = force_numerical_fit

        Distribution.__init__(self, **kwargs)

        # We also might want to warn if we have an unbounded power law (no
        # xmax) with an exponent less than 1.1, since things start to
        # get very messy around there.
        if not self.xmax and self.alpha <= 1.1:
            warnings.warn('Power law distributions with alpha close to 1 without an xmax can be very noisy; it is recommended to give some xmax.')


    def generate_initial_parameters(self, data):
        r"""
        Generate initial guesses for the distribution parameters based
        on the data.

        For continuous distributions, we use the following value to estimate
        :math:`\alpha` (see Clauset et al. (2009) [1], Eq. 3.1). In the limit as N
        goes to infinity, this becomes the exact solution to the maximum
        likelihood estimation problem. Generally this is a very good
        estimation even at modest values (~1000) of N as well.

            $$ \alpha_0 = 1 + N / ( \sum \log (x / x_{min})) $$

        For the discrete case, there is no form that is exact in the large
        N limit, but the following value is a good approximation when
        there is no ``xmax`` and ``xmin >= 10`` (see ref [1], Eq. 3.7).

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
        # If estimate_discrete is None, we decied whether to use it based
        # xmin and xmax values.
        # (we can't put this in __init__ because xmin and xmax aren't
        # defined yet)
        if self.estimate_discrete is None:
            self.estimate_discrete = (self.xmin >= 10) and (self.xmax is None) and self.discrete

        # Also give a warning if estimate discrete is true but xmin is small
        if self.estimate_discrete and (self.xmin <= 6) and self.discrete:
            warnings.warn(f'estimate_discrete=True but xmin is quite small ({self.xmin}). This may give inaccurate results.')

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

        # If we have some non-clean distributions (eg. a flat distribution that
        # then becomes power law at some xmin), the estimate above
        # could be less than 1 or even negative. It's better to just
        # start the fitting at 1 instead in those cases.
        if params["alpha"] < 1:
            params["alpha"] = 1

        return params


    def fit(self, data=None):
        r"""
        Fits the parameters of the distribution to the data.

        Overloaded from ``Distribution`` version since we may want to
        use the analytic expression for the maximum likihood estimation
        of :math:`\alpha` instead of numerically fitting.
        """
        # The generate_initial_parameters() function has already been
        # called by the time this is called, so the value of self.alpha
        # will already be set using the appropriate analytical expression.

        # Wherever we don't call Distribution.fit(), we need to call
        # compute_distance_metrics(), since this is normally done
        # at the end of the super class fit().

        # If the user has specifically requested that a numerical fit is
        # performed (using the analytical expressions as initial guesses)
        # then we just call the super method. The analytical methods are
        # already used in ``generate_initial_parameters()``, so this will
        # use those as an initial guess.
        if self.force_numerical_fit:
            return Distribution.fit(self, data)
           
        # The condition for the below two to be used is that the alpha
        # value (the true one) is within (1, 3]. That being said, the
        # estimations in generate_initial_parameters will give alpha
        # values of [1, 1.15] for power laws that actually have an alpha
        # < 1. So we should run the numerical fitting just in case even
        # if the alpha is above one, up to about 1.5 or so. By setting this
        # cutoff higher, we only lose a bit of speed, but setting it too
        # low can give inaccurate results without warning, so better to be
        # conservative here.

        # If we want to use the discrete estimation and the initial
        # guess is within (1.5, 3], we skip numerical fitting.
        if self.discrete and self.estimate_discrete and (self.alpha > 1.5) and (self.alpha <= 3):
            self.compute_distance_metrics()
            return

        # If we want to use the continuous estimation and the initial
        # guess is within  (1.5, 3], we skip numerical fitting.
        if (not self.discrete) and (self.alpha > 1.5) and (self.alpha <= 3):
            self.compute_distance_metrics()
            return

        # Otherwise we run the numerical fitting
        return Distribution.fit(self, data)


    @property
    def sigma(self):
        """
        The standard error of the MLE.
        """
        # Only is calculable after self.fit is started, when the number of data points is
        # established
        # If we try to calculate it before then (ie. when self.n doesn't
        # exist yet) we should return None. This is needed, eg. for
        # comparing pickled objects.
        if not hasattr(self, 'n'):
            return None

        return (self.alpha - 1) / np.sqrt(self.n)


    def _cdf_base_function(self, x):
        # For alpha = 1 exactly, the cdf has a logarithmic form instead
        # of another power law. Unfortunately, we can't usually tell if
        # something has an exact exponent of 1 without already calling
        # this function several times.
        if self.discrete:
            from scipy.special import zeta
            CDF = 1 - zeta(self.alpha, x)
        else:
            #Can this be reformulated to not reference xmin? Removal of the probability
            #before xmin and after xmax is handled in Distribution.cdf(), so we don't
            #strictly need this element. It doesn't hurt, for the moment.
            CDF = 1 - (x / self.xmin)**(-self.alpha + 1)
            #CDF = x**(-self.alpha + 1) - self.xmin**(-self.alpha + 1)
        return CDF


    def _pdf_base_function(self, x):
        return x**(-self.alpha)


    # TODO: This function sometimes gives its warning during the fitting
    # process, and then the final fit ends up being > 1, so it looks like
    # it raised the warning incorrectly. Maybe there is some way to clean
    # up when this is shown or not.
    @property
    def _pdf_continuous_normalizer(self):
        # The pdf has a different form when we consider xmax as
        # the upper limit of the distribution. When alpha < 1, we have to
        # assume the distribution ends at xmax otherwise it is not
        # normalizable.
        xmax = None
        if self.xmax:
            xmax = self.xmax

        else:
            if self.alpha <= 1:
                # If we have data available, we should use the maximum value and warn.
                if hasattr(self.data, '__iter__'):
                    if self.verbose:
                        warnings.warn('Distribution with alpha <= 1 has no xmax; setting xmax to be max(data) otherwise cannot continue. Consider setting an explicit value for xmax')

                        xmax = np.max(self.data)

                # Otherwise, we need to raise an error, since we cant have
                # a distribution with alpha <= 1, no xmax, and no data to
                # infer an xmax from.
                else:
                    raise ValueError('Power law distribution with alpha <= 1 must have an xmax or at least data to infer an implicit xmax from.')


        if xmax:
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
        # This estimation only works for alpha > 1
        # So even if we explicitly pass estimate_discrete=True, we can't
        # use this when alpha <= 1.
        if self.alpha <= 1:
            raise ValueError('Discrete power law random generator estimation only works for alpha > 1! Use estimate_discrete=False.')

        x = (self.xmin - 0.5) * (1 - r) ** (-1/(self.alpha - 1)) + 0.5
        return np.around(x)


class Exponential(Distribution):
    r"""
    An exponential distribution :math:`p(x) \sim e^{- \lambda x}`.

    For expressions for normalization for discrete and continuous
    distributions, see Clauset et al. (2009) [1], Table 2.1.

    References
    ----------
    [1] Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).
    Power-law distributions in empirical data. SIAM Review, 51(4),
    661–703. https://doi.org/10.1137/070710111
    """
    name = 'exponential'

    # Note that Lambda has a capital L; I guess this is because
    # the authors didn't want to cause confusion with Python's lambda
    # expressions.
    parameter_names = ['Lambda']

    DEFAULT_PARAMETER_RANGES = {'Lambda': [0, None]}


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

        """
        return np.exp(-self.Lambda * x)


    @property
    def _pdf_continuous_normalizer(self):
        # We could put an expression including xmax here but it probably
        # wouldn't make much of a difference.
        
        # There's a reasonable chance this line can lead to overflow errors
        # since it is a positive exponential with what could be a large
        # number. The end calculation would theoretically be fine since
        # the x values will always be > xmin, and therefore we will also
        # have a tiny _pdf_base_function value.

        # But in order to store this value in the meantime, we have to
        # use a float128 (longdouble) type. The ideal solution might be to use a proper
        # infinite precision module like mpmath or decimal, but I think
        # this should work for all cases.
        return self.Lambda * np.exp(np.longdouble(self.Lambda * self.xmin))


    @property
    def _pdf_discrete_normalizer(self):
        # Note that we use float128 (long double) here since otherwise
        # we might get an overflow error. See _pdf_continuous_normalizer
        # for full discussion.
        C = (1 - np.exp(-self.Lambda)) * np.exp(np.longdouble(self.Lambda * self.xmin))

        if self.xmax:
            Cxmax = (1 - np.exp(-self.Lambda)) * np.exp(np.longdouble(self.Lambda * self.xmax))
            C = 1.0/C - 1.0/Cxmax
            C = 1.0/C

        return C


    # This function used to overload the pdf() defined in the super class
    # but it doesn't seem to do anything different...?
    # If the if condition evaluates true, then it just calculated the
    # likelihood as _pdf_base_function(x) * _pdf_continuous_normalizer,
    # which is exactly what the parent pdf() does.
    def pdf(self, data=None):
        if data is None and self.parent_Fit:
            data = self.parent_Fit.data

        if not self.discrete and self.in_range() and not self.xmax:
            print('special pdf2')
            data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
            from numpy import exp
        #likelihoods = exp(-Lambda*data)*\
                #Lambda*exp(Lambda*xmin)

            # This is _pdf_base_function(x) * _pdf_continuous_normalizer(xmin)
            likelihoods = self.Lambda*exp(self.Lambda*(self.xmin-data))

            #Simplified so as not to throw a nan from infs being divided by each other
            from sys import float_info
            likelihoods[likelihoods==0] = 10**float_info.min_10_exp

        else:
            likelihoods = Distribution.pdf(self, data)

        return likelihoods

    def loglikelihoods(self, data=None):
        if data is None and self.parent_Fit:
            data = self.parent_Fit.data

        if not self.discrete and self.in_range() and not self.xmax:
            data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)
            from numpy import log
        #likelihoods = exp(-Lambda*data)*\
        #        Lambda*exp(Lambda*xmin)
            loglikelihoods = log(self.Lambda) + (self.Lambda*(self.xmin-data))
            #Simplified so as not to throw a nan from infs being divided by each other
            from sys import float_info
            loglikelihoods[loglikelihoods==0] = log(10**float_info.min_10_exp)
        else:
            loglikelihoods = Distribution.loglikelihoods(self, data)
        return loglikelihoods


    def _generate_random_continuous(self, r):
        return self.xmin - (1/self.Lambda) * np.log(1 - r)


class Stretched_Exponential(Distribution):
    r"""
    A stretched exponential distribution, :math:`p(x) \sim (x \lambda)^{\beta - 1}
    e^{- (\lambda x)^\beta}`.

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

    name = 'stretched_exponential'

    parameter_names = ['Lambda', 'beta']

    DEFAULT_PARAMETER_RANGES = {'Lambda': [0, None],
                                'beta': [0, None]}


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
        CDF = 1 - np.exp(-(self.Lambda * x)**self.beta)
        return CDF


    def _pdf_base_function(self, x):
        # TODO: This is different from the base function defined in Clauset
        # et al. 2009. It has extra factors of lambda and exp(lambda^beta)
        from numpy import exp
        return (((x*self.Lambda)**(self.beta-1)) *
                exp(-((self.Lambda*x)**self.beta)))


    @property
    def _pdf_continuous_normalizer(self):
        # Same issue here as with Exponential; we could get an overflow
        # error since this value might be very large, so we use float128.
        C = self.beta * self.Lambda * np.exp(np.longdouble(self.Lambda * self.xmin)**self.beta)
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
    r"""
    A power law distribution truncated by an exponential, :math:`p(x) \sim
    x^{-\alpha} e^{-\lambda x}`.

    The exponent :math:`\alpha` should be positive, though can have any
    positive value and still be normalizable. Unlike a true power law,
    there is no trouble having :math:`\alpha \le 1`, since the
    exponential tail ensures that the distribution is always normalizable.
    """
    name = 'truncated_power_law'

    parameter_names = ['alpha', 'Lambda']

    DEFAULT_PARAMETER_RANGES = {'alpha': [0, 3],
                                'Lambda': [0, None]}


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
        # Update: I think this approximation only applies for a true
        # power law, so it is better not to use it here.
        #if self.discrete and self.estimate_discrete and not self.xmax:
        #    params["alpha"] = 1 + n / np.sum(np.log(data / (self.xmin - 0.5)))
        #
        #else:
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
        # TODO: Try to find a way to do this without rejection sampling.
        def helper(r):
            from numpy import log
            from numpy.random import rand
            while 1:
                x = self.xmin - (1/self.Lambda) * log(1-r)
                p = ( x/self.xmin )**-self.alpha
                if rand()<p and (x >= self.xmin) and (not self.xmax or x <= self.xmax):
                    return x
                r = rand()
        from numpy import array
        return array(list(map(helper, r)))


class Lognormal(Distribution):
    r"""
    A lognormal distribution, :math:`p(x) \sim x^{-1} e^{-(\log(x) - \mu)^2 / 2 \sigma^2}`.

    Note that :math:`\sigma` is referred to as `width` in the package
    so as to not confuse it with the standard error of other distributions.
    """

    name = 'lognormal'

    # I have renamed this to be width from 'sigma' since there is
    # already a sigma defined as the standard error in this package.
    parameter_names = ['mu', 'width']

    DEFAULT_PARAMETER_RANGES = {'mu': [None, None],
                                'width': [0, None]}


    def generate_initial_parameters(self, data):
        r"""
        Generate initial guesses for the distribution parameters based
        on the data.

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

        logdata = np.log(data)

        params["mu"] = np.mean(logdata)
        params["width"] = np.std(logdata)

        return params


    def pdf(self, data=None):
        """
        Returns the probability density function (normalized histogram) of the
        theoretical distribution for the values in data within xmin and xmax,
        if present.

        This function is reimplemented solely to call _round_discrete_approx
        when we want to use that approximation. See that function (or the
        version of cdf defined in this class, which uses the same technique)
        for more information, but the goal of that function is to compute the
        cdf without normalizing until the very end.

        As of now, I can't see any reason why this would ever lead to an
        underflow error... maybe I'm missing something?

        Here's a minimal example that leads to the overflow error without
        this function:

        ```

        np.random.seed(0)
        dist = powerlaw.Lognormal(xmin=1e1, xmax=1e6, parameters=[1.5, 3], discrete=False)

        data = dist.generate_random(50000)
        powerlaw.plot_pdf(data)
        dist.plot_pdf()

        fit = powerlaw.Fit(data, xmin=1e1, xmax=1e6)

        fit.lognormal.plot_pdf()
        plt.show()

        ```

        The fitting will not converge unless you have this overloaded function.

        Parameters
        ----------
        data : list or array, optional
            If not provided, attempts to use the data from the Fit object in
            which the Distribution object is contained.

        Returns
        -------
        probabilities : array
        """
        # TODO clean this up
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

            # This is the only part I've cleaned up so far since I changed
            # the name of discrete_approximation to discrete_normalization.
            elif self.discrete_normalization == 'round':
                likelihoods = self._round_discrete_approx(data)
            elif self.discrete_normalization == 'sum':
                if self.xmax:
                    upper_limit = self.xmax
                else:
                    upper_limit = np.max(self.data)

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

        This calculation was originally reformulated to avoid underflow,
        hence why this class redefines the pdf() function. The main
        difference is that this computation (q(upper) - q(lower)) / norm, where
        q is the **unnormalized** _cdf_base_function and norm is the
        normalization constant, instead of C(upper) - C(lower), where C is the
        already normalized function. For comparison, see the regular
        cdf() function when using discrete_normalization='round'.
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
        arg1 = (np.log(lower_data)-self.mu) / (np.sqrt(2)*self.width)
        arg2 = (np.log(upper_data)-self.mu) / (np.sqrt(2)*self.width)
        likelihoods = 0.5*(ss.erfc(arg1) - ss.erfc(arg2))
        if not self.xmax:
            norm = 0.5*ss.erfc((np.log(self.xmin)-self.mu) / (np.sqrt(2)*self.width))
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
        xmax, if present.

        This function was added to reformulate the calculation to avoid
        underflow errors, though I think this isn't necessary anymore.
        As best as I can understand it, the issue was likely with some
        problems with erf, meaning that erfc was more accurate. But a better
        solution seems to be just to redefine _cdf_base_function in terms
        of erfc instead of erf, because then we don't need to redefine this
        method. I'v made that change in v1.6.0 but I will leave this function
        here for now; eventually it should be removed.

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
        import scipy.special as ss

        # The passed data takes precedence, but otherwise we use the data
        # stored in the class instance.
        if not hasattr(data, '__iter__'):
            data = self.data

        # self.data is already trimmed, but if we are passed new data
        # we need to trim it.
        data = trim_to_range(data, xmin=self.xmin, xmax=self.xmax)

        n = len(data)

        if n == 0:
            raise Exception('No data points in defined range of the distribution.')

        # If we aren't in range, we just return a bunch of (nearly) zeros
        if not self.in_range():
            return np.tile(10**sys.float_info.min_10_exp, n)

        #if self._cdf_xmin == 1:
        #    # If cdf_xmin is 1, it means we don't have the numerical accuracy to
        #    # calculate this tail. So we make everything 1, indicating
        #    # we're at the end of the tail. Such an xmin should be thrown
        #    # out by the KS test.
        #    CDF = np.ones(n)
        #    return CDF

        # Compared to the regular cdf function which uses _cdf_base_function,
        # the difference here is that we evaluate 0.5*erfc(-arg) instead of
        # 1 - 0.5*erfc(arg)

        # This is the negative of the argument for the erfc for the data
        val_data = (np.log(data) - self.mu) / (np.sqrt(2)*self.width)

        # This is the negative of the argument for the erfc in _cdf_xmin
        val_xmin = (np.log(self.xmin) - self.mu) / (np.sqrt(2)*self.width)

        # This is (1 - q(xmin)) - (1 - q(x)) = q(x) - q(xmin)
        # where q is the unnormalized CDF, ie. _cdf_base_function
        # TODO: There was an issue here with val_xmin and val_data being
        # longdouble types, so to be safe, we case here. I should try to
        # figure out where this is coming from.
        CDF = 0.5 * (ss.erfc(np.float64(val_xmin)) - ss.erfc(val_data.astype(np.float64)))

        # This is 1 - q(xmin) (because the argument is negative)
        norm = 0.5 * ss.erfc(np.float64(val_xmin))

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


    def _cdf_base_function(self, x):
        r"""
        :math:`c(x) \sim \frac{1}{2} \left( 1 + \erf \left( \frac{\log x - \mu}{\sigma \sqrt{2}} \right) \right)`
        :math:` = \frac{1}{2} \erfc \left( - \frac{\log x - \mu}{\sigma \sqrt{2}} \right)`
        """
        #from scipy.special import erfc
        #from mpmath import erfc
        #return np.float64(0.5 * erfc(-(np.log(x) - self.mu) / (self.width * np.sqrt(2))))

        from numpy import sqrt, log
        from scipy.special import erf
        return  0.5 + ( 0.5 *
                erf((log(x)-self.mu) / (sqrt(2)*self.width)))


    def _pdf_base_function(self, x):
        r"""
        :math:`p(x) \sim x^{-1} e^{-(\log(x) - \mu)^2 / 2 \sigma^2}`
        """
        #return 1/x * np.exp(-(np.log(x) - self.mu)**2 / (2 * self.width**2))
        from numpy import exp, log
        return ((1.0/x) *
                exp(-( (log(x) - self.mu)**2 )/(2*self.width**2)))


    @property
    def _pdf_continuous_normalizer(self):
        r"""
        :math:`C = \sigma \sqrt{\pi / 2} \left( 1 + \erf\left( \frac{\mu - \log x_{min}}{ \sigma \sqrt{2}} \right) \right)`
        :math:` = \sigma \sqrt{\pi / 2} \erfc\left( - \frac{\mu - \log x_{min}}{ \sigma \sqrt{2}} \right)`
        """
        #from scipy.special import erfc
        #from mpmath import erfc
        #return np.float64(self.width * np.sqrt(np.pi/2) * erfc(-(self.mu - np.log(self.xmin)) / (self.width*np.sqrt(2))))

        from mpmath import erfc
#        from scipy.special import erfc
        from scipy.constants import pi
        from numpy import sqrt, log
        C = (erfc((log(self.xmin) - self.mu) / (sqrt(2) * self.width)) /
             sqrt(2/(pi*self.width**2)))
        return float(C)


    @property
    def _pdf_discrete_normalizer(self):
        return False


    def _generate_random_continuous(self, r):
        """
        An old implementation of this used mpmath for the calculations
        and mentioned some overflow errors, but I've never encountered
        any in all of my testing. But just in case we encounter that in
        the future, we might consider switching back.
        """
        from scipy.special import erf, erfinv

        # To switch back to mpmath, uncomment these two lines
        #from mpmath import erf, erfinv
        #erfinv = np.frompyfunc(erfinv,1,1)

        erfinv_arg = erf((self.mu - np.log(self.xmin)) / (np.sqrt(2) * self.width)) * (1 - r) - r

        # To switch back to mpath, use this line instead
        #return np.exp(self.mu - np.sqrt(2) * self.width * erfinv(erfinv_arg).astype('float')

        return np.exp(self.mu - np.sqrt(2) * self.width * erfinv(erfinv_arg))


class Lognormal_Positive(Lognormal):
    r"""
    A lognormal distribution with strictly positive :math:`\mu`, :math:`p(x)
    \sim x^{-1} e^{-(\log(x) - \mu)^2 / 2 \sigma^2}`.

    """
    name = 'lognormal_positive'

    # I have renamed this to be width from 'sigma' since there is
    # already a sigma defined as the standard error in this package.
    parameter_names = ['mu', 'width']

    DEFAULT_PARAMETER_RANGES = {'mu': [0, None],
                                'width': [0, None]}
