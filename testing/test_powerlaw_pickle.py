# -*- coding: utf-8 -*-
from __future__ import (print_function, absolute_import,
                        unicode_literals, division)

from numpy.testing import assert_allclose

import walkobj as wo

import unittest
import powerlaw
import pickle

references = {
        'words': {
            'discrete': True,
            'data': powerlaw.load_test_dataset('words'),
            'alpha': 1.95,
            'xmin': 7,
            'lognormal': (0.395, 0.69),
            'exponential': (9.09, 0.0),
            'stretched_exponential': (4.13, 0.0),
            'truncated_power_law': (-0.899, 0.18),
            },
        'blackouts': {
            'discrete': False,
            'data': powerlaw.load_test_dataset('blackouts')/10.0**3,
            'alpha': 2.3,
            'xmin': 230,
            'lognormal': (-0.412, 0.68),
            'exponential': (1.43, 0.15),    # Clauset value is (1.21, 0.23),
            'stretched_exponential': (-0.417, 0.68),
            'truncated_power_law': (-0.382, 0.38),
            },
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
    'blackouts': {}
        }


class FirstTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for k in references.keys():
            data = references[k]['data']
            fit = powerlaw.Fit(data, discrete=references[k]['discrete'],
                               estimate_discrete=False)
            results[k]['alpha'] = fit.alpha
            results[k]['xmin'] = fit.xmin
            results[k]['fit'] = fit

    def test_power_pickle_dump(self):
        print("Testing power law object pickle")

        for k in references.keys():
            print(f"Test pickling of {k}")
            fname = k + "-test.pkl"
            with open(fname, "wb") as fid:
                pickle.dump(results[k]['fit'], fid)
            # print("I dumped")
            # results[k]['fit'].supported_distributions.keys()

    def test_power_pickle_load(self):
        print("Testing power law object unpickle")

        for k in references.keys():
            print(f"Test unpickling of {k}")
            fname = k + "-test.pkl"
            with open(fname, "rb") as fid:
                fitres = pickle.load(fid)
            assert wo.typedtree(fitres) == wo.typedtree(results[k]['fit'])

if __name__ == '__main__':
    # execute all TestCases in the module
    unittest.main()
