r"""
This test file contains tests that involve generating synthetic data and
then fitting to make sure we get the expected result for power law
distributions.

We generate power law data using inverse sampling transform; so essentially
the same process as is used internally, but the idea here is to compare
to a generation function that isn't defined internally, so we can isolate
issues with random generation from fitting.
"""
import powerlaw

import pytest
import unittest

import numpy as np
from numpy.testing import assert_allclose


def randomPowerLaw(alpha, xmin, xmax, size=1):
    """
    Power-law gen for:
        pdf(x) ~ x^{alpha} for xmin <= x <= xmax

    Note that this form is slightly different than that of 
    ``powerlaw.Power_Law._generate_random_continuous()`` since it is
    derived using an explicit value for xmax instead of infinity.
    """
    r = np.random.uniform(0, 1, size=size)

    return (r*xmax**(1 + alpha) + (1 - r)*xmin**(1 + alpha))**(1/(1 + alpha))


def power_law_fit(alpha_range, discrete=False):
    """
    Test power law fits on synthetic data within a specific range.
    """
    rtol = .1
    atol = 0.01

    # These are chosen arbitrarily.
    alphaArr = np.linspace(alpha_range[0], alpha_range[1],
                           int((alpha_range[1] - alpha_range[0])*15))
    numSamples = 5 # per alpha value
    N = 3000 # num samples per test

    fitAlphaArr = np.zeros((len(alphaArr), numSamples))

    # Higher xmin value for discrete since it can cause errors
    xmin = 1 if not discrete else 100

    for i in range(len(alphaArr)):
        for j in range(numSamples):
            data = randomPowerLaw(-alphaArr[i], xmin=xmin, xmax=1e6, size=N)
            if discrete:
                data = data.astype(np.int64)

            fit = powerlaw.Fit(data=data, xmin=xmin, xmax=np.max(data), verbose=0, discrete=discrete)

            fitAlphaArr[i,j] = fit.power_law.alpha


    for i in range(len(alphaArr)):
        assert_allclose(fitAlphaArr[i], np.repeat(alphaArr[i], numSamples),
                        rtol=rtol, atol=atol, err_msg=f'Alpha value: {alphaArr[i]}')


class TestExternalPowerLaw(unittest.TestCase):
    """
    These tests compare the fitting of synthetic power law data generated
    outside of the powerlaw library.
    """

    def test_alpha_1p0_to_1p5_continuous(self):
        # Can't do exactly 1.0 beacuse there is a dicontinuity there.
        power_law_fit([1.02, 1.5], discrete=False)

    def test_alpha_1p5_to_2p5_continuous(self):
        power_law_fit([1.5, 2.5], discrete=False)

    def test_alpha_2p5_to_3p0_continuous(self):
        power_law_fit([2.5, 3.0], discrete=False)

    def test_alpha_1p0_to_1p5_discrete(self):
        # Can't do exactly 1.0 beacuse there is a dicontinuity there.
        power_law_fit([1.02, 1.5], discrete=True)

    def test_alpha_1p5_to_2p5_discrete(self):
        power_law_fit([1.5, 2.5], discrete=True)

    def test_alpha_2p5_to_3p0_discrete(self):
        power_law_fit([2.5, 3.0], discrete=True)


if __name__ == '__main__':
    unittest.main()

