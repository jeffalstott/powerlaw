"""
This class tests that you can properly pickle fit objects.

The old version used the reference datasets, but I think this isn't necessary;
it is totally reasonable just to use synthetic data, since this is cleaner
and requires less overhead (making this file easier to read).
"""
import numpy as np

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
