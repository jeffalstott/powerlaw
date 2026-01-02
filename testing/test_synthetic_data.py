"""
This test file contains tests that involve generating synthetic data (using
internal functions from powerlaw) and then fitting to make sure we get the
expected result.

So a full test of most aspects of the library; these tests are meant to
ensure that the library works as it would likely be used. That being said,
these tests are probably not very helpful for debugging where an issue
is coming from, since they involve so many parts of the library.

Some general notes:

- Discrete random number generation (without an estimation method) takes
much longer than the continuous equivalent, so tests for discrete
distributions usually are repeated fewer times.

- All tests are repeated for both continuous and discrete distributions,
as well as distributions with and without an xmax value.
"""

import unittest
import pytest

import powerlaw

import numpy as np
from numpy.testing import assert_allclose
import itertools

# These are the standard choices for the tests below unless mentioned
# otherwise. There's nothing particular about these values, but they seem
# to work fine. Note that they are slightly high for testing, but we are
# generating random numbers, fitting, then generating more random numbers,
# so we want to be a little generous with the amount of error we might see.
A_TOL = 0.05
R_TOL = 0.1

def generate_fit_generate_fit(parameters_list,
                              distribution,
                              xrange=[1, 1e6],
                              discrete=False,
                              compare_log_parameter=None,
                              atol=A_TOL,
                              rtol=R_TOL):
    """
    Generate data from a distribution based on given parameters, fit the
    random data, then generate random data from the fit, and fit the new
    data again.

    Primarily a test of the random generation, but of course also tests
    fitting and creating distributions from fixed parameter values.
    """
    num_samples = 1 # repeats per set of parameter values
    N = 3000 # number of random numbers to generate and fit

    num_params = len(parameters_list[0])

    # We set the array size based on the number of parameters in the distribution
    fit_parameters_arr = np.zeros((len(parameters_list), num_samples, len(distribution.parameter_names)))
    new_fit_parameters_arr = np.zeros((len(parameters_list), num_samples, len(distribution.parameter_names)))

    for i in range(len(parameters_list)):
        for j in range(num_samples):
            # Generate data
            theoretical_dist = distribution(xmin=xrange[0], xmax=xrange[1], parameters=parameters_list[i], discrete=discrete)
            data = theoretical_dist.generate_random(N)

            if discrete:
                data = data.astype(np.int64)

            # Fit data
            fit = powerlaw.Fit(data=data, xmin=xrange[0], verbose=0, discrete=discrete)

            # Generate new data from the fit
            new_data = getattr(fit, distribution.name).generate_random(N)

            if discrete:
                new_data = new_data.astype(np.int64)

            # Fit the new data
            new_fit = powerlaw.Fit(data=new_data, xmin=xrange[0], verbose=0, discrete=discrete)

            # Get the fitted parameters
            parameters = getattr(fit, distribution.name).parameters
            new_parameters = getattr(new_fit, distribution.name).parameters
            fit_parameters_arr[i,j] = list(parameters.values())
            new_fit_parameters_arr[i,j] = list(new_parameters.values())

    # Get parameter names from the distribution
    fit_parameter_names = list(getattr(fit, distribution.name).parameters.keys())


    # If we want to compare the log of the parameters (if say we only expect
    # to fit to the order of magnitude) we should take the log of those values
    if not hasattr(compare_log_parameter, '__iter__'):
        compare_log_parameter = [False]*num_params 

    for i in range(len(parameters_list)):
        print(parameters_list[i])

        for j in range(num_params):

            if compare_log_parameter[j]:
                assert_allclose(np.log(fit_parameters_arr[i,:,j]), np.repeat(np.log(parameters_list[i][j]), num_samples),
                                rtol=rtol, atol=atol, err_msg=f'{fit_parameter_names[j]}: 1st fit, {parameters_list[i][j]}')
                
                # More tolerance for the second fit since errors could
                # propagate.
                assert_allclose(np.log(new_fit_parameters_arr[i,:,j]), np.repeat(np.log(parameters_list[i][j]), num_samples),
                                rtol=rtol*1.5, atol=atol*1.5, err_msg=f'{fit_parameter_names[j]}: 2nd fit, {parameters_list[i][j]}')

            else:
                assert_allclose(fit_parameters_arr[i,:,j], np.repeat(parameters_list[i][j], num_samples),
                                rtol=rtol, atol=atol, err_msg=f'{fit_parameter_names[j]}: 1st fit, {parameters_list[i][j]}')

                # More tolerance for the second fit since errors could
                # propagate.
                assert_allclose(new_fit_parameters_arr[i,:,j], np.repeat(parameters_list[i][j], num_samples),
                                rtol=rtol*1.5, atol=atol*1.5, err_msg=f'{fit_parameter_names[j]}: 2nd fit, {parameters_list[i][j]}')


class TestGenerationFitting_PowerLaw(unittest.TestCase):

    #######################################################################
    #                   CONTINUOUS POWER LAW
    #######################################################################

    def test_power_law_1p0_to_1p5_continuous(self):

        # Can't do exactly 1.0 beacuse there is a dicontinuity there.
        alpha_arr = np.linspace(1.02, 1.5, 5)
        parameters_list = [[a] for a in alpha_arr]

        generate_fit_generate_fit(parameters_list,
                                  distribution=powerlaw.Power_Law,
                                  xrange=[1, 1e10],
                                  discrete=False,
                                  atol=A_TOL, rtol=R_TOL)

    def test_power_law_1p5_to_2p5_continuous(self):

        alpha_arr = np.linspace(1.5, 2.5, 5)
        parameters_list = [[a] for a in alpha_arr]

        generate_fit_generate_fit(parameters_list,
                                  distribution=powerlaw.Power_Law,
                                  xrange=[1, 1e10],
                                  discrete=False,
                                  atol=A_TOL, rtol=R_TOL)

    def test_power_law_2p5_to_3p0_continuous(self):

        alpha_arr = np.linspace(2.5, 3.0, 5)
        parameters_list = [[a] for a in alpha_arr]

        generate_fit_generate_fit(parameters_list,
                                  distribution=powerlaw.Power_Law,
                                  xrange=[1, 1e10],
                                  discrete=False,
                                  atol=A_TOL, rtol=R_TOL)

    #######################################################################
    #                   DISCRETE POWER LAW
    #######################################################################

    def test_power_law_1p0_to_1p5_discrete(self):

        # Can't do exactly 1.0 beacuse there is a dicontinuity there.
        alpha_arr = np.linspace(1.02, 1.5, 5)
        parameters_list = [[a] for a in alpha_arr]

        generate_fit_generate_fit(parameters_list,
                                  distribution=powerlaw.Power_Law,
                                  xrange=[5e1, 1e10],
                                  discrete=True,
                                  atol=A_TOL, rtol=R_TOL)

    def test_power_law_1p5_to_2p5_discrete(self):

        alpha_arr = np.linspace(1.5, 2.98, 5)
        parameters_list = [[a] for a in alpha_arr]

        generate_fit_generate_fit(parameters_list,
                                  distribution=powerlaw.Power_Law,
                                  xrange=[5e1, 1e10],
                                  discrete=True,
                                  atol=A_TOL, rtol=R_TOL)

    def test_power_law_2p5_to_3p0_discrete(self):

        # We do slightly less than 3 because the default bound is at 3,
        # and we want to test this without changing bounds.
        alpha_arr = np.linspace(2.5, 2.98, 5)
        parameters_list = [[a] for a in alpha_arr]

        generate_fit_generate_fit(parameters_list,
                                  distribution=powerlaw.Power_Law,
                                  xrange=[5e1, 1e10],
                                  discrete=True,
                                  atol=A_TOL, rtol=R_TOL)

class TestGenerationFitting_Exponential(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lambda_arr = np.logspace(-4, -2, 3)
        cls.parameters_list = [[l] for l in lambda_arr]


    #######################################################################
    #                   CONTINUOUS EXPONENTIAL
    #######################################################################

    def test_exponential_continuous(self):
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Exponential,
                                  xrange=[1, None],
                                  discrete=False,
                                  compare_log_parameter=[True],
                                  atol=A_TOL, rtol=R_TOL)

    def test_exponential_continuous_xmax(self):
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Exponential,
                                  xrange=[1, 1e5],
                                  discrete=False,
                                  compare_log_parameter=[True],
                                  atol=A_TOL, rtol=R_TOL)


    #######################################################################
    #                   DISCRETE EXPONENTIAL
    #######################################################################

    def test_exponential_discrete(self):
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Exponential,
                                  xrange=[1, None],
                                  discrete=True,
                                  compare_log_parameter=[True],
                                  atol=A_TOL, rtol=R_TOL)


    def test_exponential_discrete_xmax(self):
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Exponential,
                                  xrange=[1, 1e5],
                                  discrete=True,
                                  compare_log_parameter=[True],
                                  atol=A_TOL, rtol=R_TOL)


class TestGenerationFitting_StretchedExponential(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lambda_arr = np.logspace(-4, -2, 3)
        beta_arr = np.linspace(0.4, 1, 3)
        cls.parameters_list = list(itertools.product(lambda_arr, beta_arr))


    #######################################################################
    #                   CONTINUOUS STRETCHED EXPONENTIAL
    #######################################################################

    def test_stretched_exponential_continuous(self):
        # Compare log(lambda) to lambda since this could span many orders
        # of magnitude in theory.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Stretched_Exponential,
                                  xrange=[1, None],
                                  discrete=False,
                                  compare_log_parameter=[True, False],
                                  atol=A_TOL, rtol=R_TOL)

    def test_stretched_exponential_continuous_xmax(self):
        # Compare log(lambda) to lambda since this could span many orders
        # of magnitude in theory.
        # And we use a slightly higher xmax since  for very small lambda
        # we need a little more room to accurately fit.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Stretched_Exponential,
                                  xrange=[1, 1e8],
                                  discrete=False,
                                  compare_log_parameter=[True, False],
                                  atol=A_TOL, rtol=R_TOL)


    #######################################################################
    #                   DISCRETE STRETCHED EXPONENTIAL
    #######################################################################

    def test_stretched_exponential_discrete(self):
        # Compare log(lambda) to lambda since this could span many orders
        # of magnitude in theory.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Stretched_Exponential,
                                  xrange=[1, None],
                                  discrete=True,
                                  compare_log_parameter=[True, False],
                                  atol=A_TOL, rtol=R_TOL)

    def test_stretched_exponential_discrete_xmax(self):
        # Compare log(lambda) to lambda since this could span many orders
        # of magnitude in theory.
        # And we use a slightly higher xmax since  for very small lambda
        # we need a little more room to accurately fit.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Stretched_Exponential,
                                  xrange=[1, 1e8],
                                  discrete=True,
                                  compare_log_parameter=[True, False],
                                  atol=A_TOL, rtol=R_TOL)


class TestGenerationFitting_TruncatedPowerLaw(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Truncated power laws have some heavy functions calls since they
        # use gamma functions and things.
        # As such, we only try a small set of parameters
        cls.parameters_list = [[1.5, 1e-3],
                               [1.2, 1e-2]]


    #######################################################################
    #                   CONTINUOUS TRUNCATED POWER LAW
    #######################################################################

    def test_truncated_power_law_continuous(self):
        # Compare log(lambda) to log(lambda) since this could span many orders
        # of magnitude in theory.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Truncated_Power_Law,
                                  xrange=[1, None],
                                  discrete=False,
                                  compare_log_parameter=[False, True],
                                  atol=A_TOL, rtol=R_TOL)

    def test_truncated_power_law_continuous_xmax(self):
        # Compare log(lambda) to log(lambda) since this could span many orders
        # of magnitude in theory.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Truncated_Power_Law,
                                  xrange=[1, 1e5],
                                  discrete=False,
                                  compare_log_parameter=[False, True],
                                  atol=A_TOL, rtol=R_TOL)


    #######################################################################
    #                   DISCRETE TRUNCATED POWER LAW
    #######################################################################

    def test_truncated_power_law_discrete(self):
        # Compare log(lambda) to log(lambda) since this could span many orders
        # of magnitude in theory.
        # Also, fitting discrete truncated power laws is difficult, not
        # exactly sure why, but we tend to have more error than other
        # distributions, so we increase the tolerances slightly.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Truncated_Power_Law,
                                  xrange=[1, None],
                                  discrete=True,
                                  compare_log_parameter=[False, True],
                                  atol=A_TOL, rtol=R_TOL*2)

    def test_truncated_power_law_discrete_xmax(self):
        # Compare log(lambda) to log(lambda) since this could span many orders
        # of magnitude in theory.
        # Also, fitting discrete truncated power laws is difficult, not
        # exactly sure why, but we tend to have more error than other
        # distributions, so we increase the tolerances slightly.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Truncated_Power_Law,
                                  xrange=[1, 1e5],
                                  discrete=True,
                                  compare_log_parameter=[False, True],
                                  atol=A_TOL, rtol=R_TOL*2)


class TestGenerationFitting_Lognormal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Lognormals can be a bit finicky depending on the parameters,
        # so it's better to just pick a few reasonable values than try
        # to sweep a range.
        cls.parameters_list = [[10, 5], # [mu, sigma]
                               [5, 1.5]]


    #######################################################################
    #                   CONTINUOUS LOGNORMAL
    #######################################################################

    def test_lognormal_continuous(self):
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Lognormal,
                                  xrange=[1, None],
                                  discrete=False,
                                  compare_log_parameter=[True, False],
                                  atol=A_TOL, rtol=R_TOL)

    def test_lognormal_continuous_xmax(self):
        # Lognormals have a higher finite xmax than other tests since
        # for these parameter values they decay much faster.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Lognormal,
                                  xrange=[1, 1e15],
                                  discrete=False,
                                  compare_log_parameter=[True, False],
                                  atol=A_TOL, rtol=R_TOL)


    #######################################################################
    #                   DISCRETE LOGNORMAL
    #######################################################################

    def test_lognormal_discrete(self):
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Lognormal,
                                  xrange=[1, None],
                                  discrete=True,
                                  compare_log_parameter=[True, False],
                                  atol=A_TOL, rtol=R_TOL)

    def test_lognormal_discrete_xmax(self):
        # Lognormals have a higher finite xmax than other tests since
        # for these parameter values they decay much faster.
        generate_fit_generate_fit(self.parameters_list,
                                  distribution=powerlaw.Lognormal,
                                  xrange=[1, 1e15],
                                  discrete=True,
                                  compare_log_parameter=[True, False],
                                  atol=A_TOL, rtol=R_TOL)


if __name__ == '__main__':
    unittest.main()
