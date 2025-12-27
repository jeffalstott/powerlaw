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


def randomPowerLawXmin(alpha, x0, xmax, size=1):
    """
    Power-law gen for:
        pdf(x) ~ const       for x0   <= x <= xmin
        pdf(x) ~ x^{alpha}   for xmin <= x <= xmax

    ie. generate a flat distribution that then decays like a power law
    so we can test the xmin fitting.

    xmin is randomly generated from a (log) uniform distribution up
    to 10**(0.1 np.log10(np.max(data))) * 0.01 is chosen so that
    way we always have at least two decades to fit the powerlaw to.

    log(xmax/x0) should probably be at least 5 to give ample range to generate
    data.
    """
    maxExp = np.log10(xmax) - 2
    xmin = 10**np.random.uniform(np.log10(x0) + 1, maxExp)
    nonPowerLawFrac = (xmin - x0) / (xmax - x0)

    r = np.random.random(size=int(size*(1 - nonPowerLawFrac)))
    xming, xmaxg = xmin**(alpha+1), xmax**(alpha+1)
    powerLawValues = (xming + (xmaxg - xming)*r)**(1./(alpha+1))
    hist, bins = np.histogram(powerLawValues, bins=np.logspace(np.log10(xmin), np.log10(xmax), 10))

    nonPowerLawSamples = (xmin - x0) / (bins[1] - bins[0]) * hist[0]
    nonPowerLawValues = np.random.uniform(x0, xmin, size=int(nonPowerLawSamples))

    return np.concatenate((nonPowerLawValues, powerLawValues)), xmin


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


def power_law_fit_random_xmin(alpha_range):
    """
    Test power law fits on synthetic data within a specific range, with a
    random xmin value, before which the distribution is flat.
    """
    # Higher tolerance, since this fit will be noisier
    rtol = .2
    atol = 0.5

    # These are chosen arbitrarily.
    alphaArr = np.linspace(alpha_range[0], alpha_range[1],
                           int((alpha_range[1] - alpha_range[0])*5))
    numSamples = 1 # per alpha value
    N = 3000 # num samples per test

    xminArr = np.zeros((len(alphaArr), numSamples))
    fitAlphaArr = np.zeros((len(alphaArr), numSamples))
    fitXMinArr = np.zeros((len(alphaArr), numSamples))

    for i in range(len(alphaArr)):
        for j in range(numSamples):
            data, xmin = randomPowerLawXmin(-alphaArr[i], x0=1, xmax=1e6, size=N)
            fit = powerlaw.Fit(data, xmax=np.max(data), verbose=0)

            xminArr[i,j] = xmin
            fitAlphaArr[i,j] = fit.power_law.alpha
            fitXMinArr[i,j] = fit.xmin

    for i in range(len(alphaArr)):
        assert_allclose(fitAlphaArr[i], np.repeat(alphaArr[i], numSamples),
                        rtol=rtol, atol=atol, err_msg=f'alpha mismatch for alpha value: {alphaArr[i]}')

        # For the xmin, just make sure the log of the value is close
        assert_allclose(np.log(fitXMinArr[i]), np.log(xminArr[i]),
                        rtol=rtol, atol=atol, err_msg=f'xmin mismatch for alpha value: {alphaArr[i]}')


class TestExternalPowerLaw(unittest.TestCase):
    """
    These tests compare the fitting of synthetic power law data generated
    outside of the powerlaw library.
    """

    # This fails because of the divergence of the pdf/cdf close to 1, see:
    # https://github.com/jeffalstott/powerlaw/issues/119
    @pytest.mark.xfail(reason="https://github.com/jeffalstott/powerlaw/issues/119")
    def test_alpha_1p0_to_1p5_continuous(self):
        # Can't do exactly 1.0 beacuse there is a dicontinuity there.
        power_law_fit([1.02, 1.5], discrete=False)

    def test_alpha_1p5_to_2p5_continuous(self):
        power_law_fit([1.5, 2.5], discrete=False)

    def test_alpha_2p5_to_3p0_continuous(self):
        power_law_fit([2.5, 3.0], discrete=False)

    def test_alpha_2p0_to_2p5_continuous_random_xmin(self):
        # These tests with xmin take much longer, so for now I'll leave
        # just a single one.
        power_law_fit_random_xmin([2.0, 2.5])

    def test_alpha_1p0_to_1p5_discrete(self):
        # Can't do exactly 1.0 beacuse there is a dicontinuity there.
        power_law_fit([1.02, 1.5], discrete=True)

    def test_alpha_1p5_to_2p5_discrete(self):
        power_law_fit([1.5, 2.5], discrete=True)

    def test_alpha_2p5_to_3p0_discrete(self):
        power_law_fit([2.5, 3.0], discrete=True)


if __name__ == '__main__':
    unittest.main()
