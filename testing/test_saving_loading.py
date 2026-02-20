"""
This class tests that you can properly pickle fit objects.

The old version used the reference datasets, but I think this isn't necessary;
it is totally reasonable just to use synthetic data, since this is cleaner
and requires less overhead (making this file easier to read).
"""
import numpy as np
from numpy.testing import assert_equal, assert_allclose

import os
import unittest
import powerlaw
import pickle

class TestHashing(unittest.TestCase):

    def test_hash_equal(self):
        """
        """
        # The specific dataset is arbitrary
        data = powerlaw.load_test_dataset('blackouts')

        # xmin given in different type (but same value)
        fit_1 = powerlaw.Fit(data, xmin=1)
        fit_2 = powerlaw.Fit(data, xmin=1.0)

        assert hash(fit_1) == hash(fit_2)

        # estimate_discrete for non discrete distribution
        fit_1 = powerlaw.Fit(data, xmin=1)
        fit_2 = powerlaw.Fit(data, xmin=1, estimate_discrete=True)

        assert hash(fit_1) == hash(fit_2)

        # Fitting xmin
        fit_1 = powerlaw.Fit(data)
        fit_2 = powerlaw.Fit(data)

        assert hash(fit_1) == hash(fit_2)

        # Data type
        fit_1 = powerlaw.Fit(data.astype(np.float32))
        fit_2 = powerlaw.Fit(data.astype(np.float64))

        assert hash(fit_1) == hash(fit_2)

        # Rounding error
        fit_1 = powerlaw.Fit(data)
        fit_2 = powerlaw.Fit(data + 1e-20)

        assert hash(fit_1) == hash(fit_2)

        # Accessing a distribution
        fit_1 = powerlaw.Fit(data, discrete=True)
        fit_2 = powerlaw.Fit(data, discrete=True)
        fit_2.power_law

        assert hash(fit_1) == hash(fit_2)


    def test_hash_not_equal(self):
        """
        """
        # The specific dataset is arbitrary
        data = powerlaw.load_test_dataset('fires')

        # Changing the data
        fit_1 = powerlaw.Fit(data, xmin=1)
        fit_2 = powerlaw.Fit(data + 1e-5, xmin=1)

        assert hash(fit_1) != hash(fit_2)

        # Changing the xmin
        fit_1 = powerlaw.Fit(data, xmin=1)
        fit_2 = powerlaw.Fit(data, xmin=2)

        assert hash(fit_1) != hash(fit_2)

        # Discrete vs continuous
        fit_1 = powerlaw.Fit(data, xmin=1, discrete=False)
        fit_2 = powerlaw.Fit(data, xmin=1, discrete=True)

        assert hash(fit_1) != hash(fit_2)

        # xmax
        fit_1 = powerlaw.Fit(data, xmin=1, xmax=100)
        fit_2 = powerlaw.Fit(data, xmin=1, xmax=None)

        assert hash(fit_1) != hash(fit_2)


TEST_FILE = 'unit_test_saving_loading.h5'

class TestSavingLoading(unittest.TestCase):

    def test_compare_hash(self):
        """
        This test makes sure that saved and loaded files have the same
        hash. This covers pretty much all of the variables in the class,
        though there are a few outside of this that we test separately.
        """
        # The specific dataset is arbitrary
        data = powerlaw.load_test_dataset('blackouts')

        # With fixed xmin
        try:
            original_fit = powerlaw.Fit(data, xmin=1)
            original_fit.save(TEST_FILE)
            loaded_fit = powerlaw.Fit.load(TEST_FILE)

            assert hash(original_fit) == hash(loaded_fit)

        finally:
            os.remove(TEST_FILE)

        # Fitting xmin
        try:
            original_fit = powerlaw.Fit(data)
            original_fit.save(TEST_FILE)
            loaded_fit = powerlaw.Fit.load(TEST_FILE)

            assert hash(original_fit) == hash(loaded_fit)

        finally:
            os.remove(TEST_FILE)


    def test_compare_xmin_fitting(self):
        """
        The xmin fitting results can't be included in the hash because
        we might not know the results at the time we need to hash, so
        we test them separately.
        """
        # The specific dataset is arbitrary
        data = powerlaw.load_test_dataset('blackouts')

        try:
            original_fit = powerlaw.Fit(data)
            original_fit.save(TEST_FILE)
            loaded_fit = powerlaw.Fit.load(TEST_FILE)

            fitting_result_keys = ['distances', 'xmins', 'valid_fits']
            for key in fitting_result_keys:
                assert_equal(original_fit.xmin_fitting_results[key], loaded_fit.xmin_fitting_results[key])

        finally:
            os.remove(TEST_FILE)


    def test_compare_fit(self):
        """
        This test compares the fitted parameters for a specific distribution
        between the original and loaded fits.

        This is quite an important test even if it doesn't seem like it. This
        is because the actual distribution objects aren't saved with the
        cahched file, so they have to be reconstructed based on the parameters
        and data. So if the distribution fitting is exactly the same, that is
        quite an important result.
        """
        # The specific dataset is arbitrary
        data = powerlaw.load_test_dataset('flares')

        try:
            original_fit = powerlaw.Fit(data)
            original_fit.save(TEST_FILE)
            loaded_fit = powerlaw.Fit.load(TEST_FILE)

            for dist in original_fit.supported_distributions.keys():
                original_values = list(getattr(original_fit, dist).parameters.values())
                loaded_values = list(getattr(loaded_fit, dist).parameters.values())

                assert_allclose(original_values, loaded_values, rtol=0.01, atol=0.01)

        finally:
            os.remove(TEST_FILE)


    def test_compare_fit_initial_parameters(self):
        """
        This test compares the fitted parameters for a specific distribution
        between the original and loaded fits.
        """
        # The specific dataset is arbitrary
        data = powerlaw.load_test_dataset('flares')

        try:
            # The parameters are intentionally scattered (and you would almost
            # never given a None value for one, but since this is for unit
            # testing we want to push things a bit).
            original_fit = powerlaw.Fit(data, initial_parameters={"alpha": 1.2, "Lambda": 1e-2, "mu": None})
            original_fit.save(TEST_FILE)
            loaded_fit = powerlaw.Fit.load(TEST_FILE)

            for dist in original_fit.supported_distributions.keys():
                original_values = list(getattr(original_fit, dist).parameters.values())
                loaded_values = list(getattr(loaded_fit, dist).parameters.values())

                assert_allclose(original_values, loaded_values, rtol=0.01, atol=0.01)

        finally:
            os.remove(TEST_FILE)


    def test_compare_fit_parameter_ranges(self):
        """
        This test compares the fitted parameters for a specific distribution
        between the original and loaded fits.
        """
        # The specific dataset is arbitrary
        data = powerlaw.load_test_dataset('flares')

        try:
            # The parameters are intentionally scattered (and you would almost
            # never given a None value for one, but since this is for unit
            # testing we want to push things a bit).
            original_fit = powerlaw.Fit(data, parameter_ranges={"alpha": [1.2, 1.5], "Lambda": [1e-5, None]})
            original_fit.save(TEST_FILE)
            loaded_fit = powerlaw.Fit.load(TEST_FILE)

            for dist in original_fit.supported_distributions.keys():
                original_values = list(getattr(original_fit, dist).parameters.values())
                loaded_values = list(getattr(loaded_fit, dist).parameters.values())

                assert_allclose(original_values, loaded_values, rtol=0.01, atol=0.01)

        finally:
            os.remove(TEST_FILE)


    def test_compare_fit_parameter_constraints(self):
        """
        This test compares the fitted parameters for all distributions with
        a constraint.
        """
        # The specific dataset is arbitrary
        data = powerlaw.load_test_dataset('blackouts')

        try:
            # As discussed in the documentation, this needs to be entirely
            # self contained.
            def constr(dist):
                N = 100
                return len(dist.data) - N

            constraint_dict = {"type": 'ineq',
                               "fun": constr,
                               "dists": ['power_law']}

            # The parameters are intentionally scattered (and you would almost
            # never given a None value for one, but since this is for unit
            # testing we want to push things a bit).
            original_fit = powerlaw.Fit(data, parameter_constraints=[constraint_dict])
            original_fit.save(TEST_FILE)
            loaded_fit = powerlaw.Fit.load(TEST_FILE)

            for dist in original_fit.supported_distributions.keys():
                original_values = list(getattr(original_fit, dist).parameters.values())
                loaded_values = list(getattr(loaded_fit, dist).parameters.values())

                assert_allclose(original_values, loaded_values, rtol=0.01, atol=0.01)

        finally:
            os.remove(TEST_FILE)
