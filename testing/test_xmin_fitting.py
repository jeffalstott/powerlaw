r"""
This test file contains tests that involve fitting the xmin value for different
distributions.

Synthetic data is generated that has a flat probability distribution up
until some xmin value, which then scales with the specified distribution.
Fitting is performed to find this xmin value, and compared to the value
for which the data is generated.
"""
import powerlaw

import pytest
import unittest

import numpy as np
from numpy.testing import assert_allclose


def random_samples_xmin(dist, x0, xmax, xmin=None, size=1):
    """
    Generate random numbers that have a flat distribution, followed
    by a region that obeys the given ``dist``.
    
        pdf(x) ~ const       for x0   <= x <= xmin
        pdf(x) ~ dist        for xmin <= x <= xmax

    If not provided, xmin is generated randomly in the given range,
    with some padding on either side to make sure that both scaling regions
    are prominent enough to fit.

    Parameters
    ----------
    dist : powerlaw.Distribution
        A distribution object that defines the scaling in the second regime.

        A copy will be made that is appropriately bounded, so it needs not
        include these bounds when passed.

    x0 : float
        The minimum value where the flat scaling regime begins.

    xmax : float
        The maximum value where the distribution's scaling regime ends.

    xmin : float, optional
        The transition point from a flat distribution to a scaling one.

        If not provided, a random one will be drawn in the appropriate domain.

    size : int, optional
        The number of random numbers to generate.
    """
    if xmin is None:
        # For an exponential distribution, we need to draw the xmin considering
        # linear x bins, so we will choose some tiny (or much too large) value if
        # we use logarithmic x bins
        if type(dist) == powerlaw.Exponential:
            # We pad the xmin drawing a bit so we don't end up with so little a
            # scaling region that we can't actually fit
            xmin = np.random.uniform(x0*5, xmax*0.5)
        else:
            maxExp = np.log10(xmax) - 2
            xmin = 10**np.random.uniform(np.log10(x0) + 1, maxExp)

    non_dist_frac = (xmin - x0) / (xmax - x0)

    # Make a copy of the distribution that obeys the xmin and xmax (just in
    # case the passed on doesn't). Note this ignores if the distribution was
    # discrete.
    bounded_dist = type(dist)(xmin=xmin, xmax=xmax, parameters=dist.parameters)
    
    dist_values = bounded_dist.generate_random(int(size*(1 - non_dist_frac)))

    # Again, if we have an exponential, this has to use linear bins
    if type(dist) == powerlaw.Exponential:
        hist, bins = np.histogram(dist_values, bins=np.linspace(xmin, xmax, 10))
        
    else:
        hist, bins = np.histogram(dist_values, bins=np.logspace(np.log10(xmin), np.log10(xmax), 10))
    
    num_non_dist_values = (xmin - x0) / (bins[1] - bins[0]) * hist[0]
    non_dist_values = np.random.uniform(x0, xmin, size=int(num_non_dist_values))

    return np.concatenate((non_dist_values, dist_values)), xmin


def distribution_fit_xmin(distribution, x0=1, xmax=1e6, xmin=None):
    """
    Test xmin fitting on synthetic data within a specific range, with a
    specified xmin value, before which the distribution is flat.

    Parameters
    ----------
    distribution : powerlaw.Distribution
        A distribution to use to generate the random samples.

    x0 : float
        The minimum value where the flat scaling regime begins.

    xmax : float
        The maximum value where the distribution's scaling regime ends.

    xmin : float, optional
        The xmin value at which to transition from a flat PDF to the
        scaling region. If not provided, a random value will be selected.
    """
    # Higher tolerance, since this fit will be noisier
    rtol = .2
    atol = 0.5

    N = 5000 # num samples per test

    data, true_xmin = random_samples_xmin(distribution, x0=x0, xmax=xmax, size=N)
    fit = powerlaw.Fit(data, xmax=np.max(data), verbose=0)

    # For the xmin, just make sure the log of the value is close
    assert_allclose(np.log10([true_xmin]), np.log10([xmin]),
                    rtol=rtol, atol=atol, err_msg=f'xmin mismatch for dist: {distribution.name}, {distribution.parameters}')


class TestXminFitting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_power_law(self):
        dist = powerlaw.Power_Law(xmin=1, alpha=1.5)
        distribution_fit_xmin(dist, x0=1, xmax=1e4, xmin=1e2)

    def test_exponential(self):
        dist = powerlaw.Exponential(xmin=1, Lambda=1e-2)
        distribution_fit_xmin(dist, x0=1, xmax=1e3, xmin=200)

    def test_stretched_exponential(self):
        dist = powerlaw.Stretched_Exponential(xmin=1, Lambda=1e-2, beta=0.7)
        distribution_fit_xmin(dist, x0=1, xmax=1e3, xmin=20)


if __name__ == '__main__':
    unittest.main()

