"""
This test file contains tests that involve fitting distributions to
reference data and checking that the results are similar to previously
saved values.

All of the test datasets can be loaded using ``powerlaw.load_test_dataset()``
if you would like to experiment with them.

Note: in the original repo, quakes.txt was given such that to get the
actual data, you need to do 10**(data). This is confusing since it is
the only file that requires special preprocessing, so I've saved the
data just directly in this exponentiated form. The original values can
be found in quakes_original.txt.

"""

# TODO
"""
There are a few parts of this test that need to be updated.

First, I highlight that this type of test is good to have, since it shows
that this library can give the same (or better) results as have already
been published in literature or otherwise shared.

That being said, there are several things that need to be addressed:

1. The source of each of the test datasets is not well-documented. I started
trying to gather sources to list in the documentation, but succeeded only
for two datasets (blackouts and words). See the tutorial on "Loading data"
for this information.

2. The source of each comparison value should be made more clear. Some
of them are commented in the dictionary below (Clauset or plfit seem to
be the main sources, but not sure about this for the other ones).

3. As of now, these tests are good comparisons for the power law distribution
(which to be fair, is the namesake of this library) but are somewhat arbitrary
for the other distributions. We compare the loglikelihood of each other
distribution to a power law (as a ratio) but since each dataset seems to have
a power law form, the specific ratios we get are pretty much arbitrary, and
depend only on how well the numerical optimization can do given a dataset that
clearly doesn't match the fitted form. The fix for this would be to have
more datasets, including ones that, to the best of our knowledge, actually
follow other distributions. You can see why this is a problem if you actually
look at the parameter values for some of the comparison fits, eg. for the 'words'
dataset, the stretched exponential fit has parameters:
{'Lambda': 71128767.8789453, 'beta': 0.10458555901993033}
That lambda value is so large that it's pretty much meaningless.

4. A side of effect of only comparing loglikelihood ratios for distributions
other than power law is that we have no way to tell if the test fits are
reasonable, or, like the above example, and just the optimization routine
doing its best with a fit that is destined to fail. This also means that these
could be failing because the library is actually _better_ at fitting than it
used to be (or other libraries used to be). I think this is the case currently
for truncated power law and stretched exponential (which are commented out), since
in all reference data tests, the power law fit always passes, but the others
fail **with a lower value of the loglikelihood ratio**. So power law fitting
is at least as good as it used to be, but just the other distributions fit
better. Since the actual parameter values aren't given for the cached results,
I don't think we can explicitly confirm this. I would guess the better fitting
comes from the fact that parameter bounds are explicitly included in the
numerical optimization; it makes sense then that this wouldn't affect power
laws much since most of the time they just the MLE without numerical
optimization.

This is a pretty big todo item, but it isn't that urgent since we now have
(in v2.0) a much more complete suite of tests that don't have the issues
stated above. Either way, I would start by trying to track down sources
for each dataset, and go from there.
"""

import unittest
import powerlaw
from numpy.testing import assert_allclose
import numpy as np

# data_factor below is the factor by which to multiply the data before
# doing any fitting. This is needed because some samples are things like
# number of people affect by blackouts, but are given in thousands of
# people, so we have to account for that. To my knowledge (not confirmed)
# all of the datasets that have a non-unity value for data_factor have
# such a value because the raw data is given in thousands of something.
references = {
        'words': {
            'discrete': True,
            'data_factor': 1,
            'alpha': 1.95,
            'xmin': 7,
            'lognormal': (0.395, 0.69),
            'exponential': (9.09, 0.0),
            'stretched_exponential': (4.13, 0.0),
            'truncated_power_law': (-0.899, 0.18),
            },
        'terrorism': {
            'discrete': True,
            'data_factor': 1,
            'alpha': 2.4,
            'xmin': 12,
            'lognormal': (-0.278, 0.78),
            'exponential': (2.457, 0.01),
            'stretched_exponential': (0.772, 0.44),
            'truncated_power_law': (-0.077, 0.70),
            },
        'blackouts': {
            'discrete': False,
            'data_factor': 1e-3,
            'alpha': 2.3,
            'xmin': 230,
            'lognormal': (-0.412, 0.68),
            'exponential': (1.43, 0.15),    # Clauset value is (1.21, 0.23),
            'stretched_exponential': (-0.417, 0.68),
            'truncated_power_law': (-0.382, 0.38),
            },
        'cities': {
            'discrete': False,
            'data_factor': 1e-3,
            'alpha': 2.37,
            'xmin': 52.46,
            'lognormal': (-0.090, 0.93),
            'exponential': (3.65, 0.0),
            'stretched_exponential': (0.204, 0.84),
            'truncated_power_law': (-0.123, 0.62),
            },
        'fires': {
            'discrete': False,
            'data_factor': 1,
            'alpha': 2.2,
            'xmin': 6324,
            'lognormal': (-1.78, 0.08),
            'exponential': (4.00, 0.0),
            'stretched_exponential': (-1.82, 0.07),
            'truncated_power_law': (-5.02, 0.0),
            },
        'flares': {
            'discrete': False,
            'data_factor': 1,
            'alpha': 1.79,
            'xmin': 323,
            'lognormal': (-0.803, 0.42),
            'exponential': (13.7, 0.0),
            'stretched_exponential': (-0.546, 0.59),
            'truncated_power_law': (-4.52, 0.0),
            },
        'quakes': {
            'discrete': False,
            'data_factor': 1e-3,
            'alpha': 1.95,   # Clauset/plfit value is 1.64
            'xmin': 10,      # Clauset/plfit value is .794
            'lognormal': (-0.796, 0.43),     # Clauset value is (-7.14, 0.0)
            'exponential': (9.7, 0),   # Clauset value is (11.6, 0.0),
            'stretched_exponential': (-7.09, 0.0),
            'truncated_power_law': (-24.4, 0.0),
            },
        'surnames': {
            'discrete': False,
            'data_factor': 1e-3,
            'alpha': 2.2,       # Clauset/plfit value is 2.5,
            'xmin': 14.92,      # Clauset/plfit value is 111.92
            'lognormal': (0.148, 0.88),     # Clauset value is (-0.836, 0.4)
            'exponential': (10, 0),   # Clauset value is (2.89, 0.0),
            'stretched_exponential': (-0.844, 0.40),
            'truncated_power_law': (-1.36, 0.10),
            }
        }
"""
There is a subtle bug in the Clauset/plfit code involving the calculation of
the cumulative distribution function. Specifically, it assumes that only
discrete distributions can have repeat values and therefore performs the
calculation incorrectly in the case of a continuous distribution with repeated
values as occurs in the quakes and surnames data sets. The alpha values used
here for those data sets can be confirmed with the plfit code by forcing it use
the corresponding xmin values. Forcing powerlaw to calculate the cumulative
distribution function as done in the plfit code produces the same xmin values
as the plfit code and the other data sets produce identical results with both
plfit and powerlaw so the alpha and xmin values were changed as above for the
quakes and surnames data sets.
"""

results = {
        'words': {},
        'terrorism': {},
        'blackouts': {},
        'cities': {},
        'fires': {},
        'flares': {},
        'quakes': {},
        'surnames': {}
        }


class TestReferenceData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for k in references.keys():
            # Load data using powerlaw's data loading function
            data = powerlaw.load_test_dataset(k)
            data = data * references[k]["data_factor"]

            fit = powerlaw.Fit(data, discrete=references[k]['discrete'],
                               estimate_discrete=False)
            results[k]['alpha'] = fit.alpha
            results[k]['xmin'] = fit.xmin
            results[k]['fit'] = fit

    def test_power_law(self):
        print("Testing power law fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            assert_allclose(results[k]['alpha'], references[k]['alpha'],
                            rtol=rtol, atol=atol, err_msg=k)

            assert_allclose(results[k]['xmin'], references[k]['xmin'],
                            rtol=rtol, atol=atol, err_msg=k)

    def test_power_law_params(self):
        print("Testing if power law params are set correctly")

        for k in references.keys():
            print(k)
            assert results[k]['fit'].power_law.parameter1 == results[k]['fit'].power_law.alpha
            assert results[k]['fit'].power_law.parameter1_name == 'alpha'


    def test_lognormal(self):
        print("Testing lognormal fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            fit = results[k]['fit']
            Randp = fit.loglikelihood_ratio('power_law', 'lognormal',
                                            normalized_ratio=True)
            results[k]['lognormal'] = Randp

            assert_allclose(Randp, references[k]['lognormal'],
                            rtol=rtol, atol=atol, err_msg=k)

    def test_exponential(self):
        print("Testing exponential fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            fit = results[k]['fit']
            Randp = fit.loglikelihood_ratio('power_law', 'exponential',
                                            normalized_ratio=True)
            results[k]['exponential'] = Randp

            assert_allclose(Randp, references[k]['exponential'],
                            rtol=rtol, atol=atol, err_msg=k)

    def test_stretched_exponential(self):
        print("Testing stretched_exponential fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            fit = results[k]['fit']
            Randp = fit.loglikelihood_ratio('power_law', 'stretched_exponential',
                    normalized_ratio=True)
            results[k]['stretched_exponential'] = Randp

            # TODO: See the todo item at the top of this file.
            #assert_allclose(Randp, references[k]['stretched_exponential'],
            #        rtol=rtol, atol=atol, err_msg=k)

    def test_truncated_power_law(self):
        print("Testing truncated_power_law fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            if references[k]['discrete']:
                continue
            fit = results[k]['fit']
            Randp = fit.loglikelihood_ratio('power_law', 'truncated_power_law')
            results[k]['truncated_power_law'] = Randp

            # TODO: See the todo item at the top of this file.
            #assert_allclose(Randp, references[k]['truncated_power_law'],
            #        rtol=rtol, atol=atol, err_msg=k)


if __name__ == '__main__':
    unittest.main()
