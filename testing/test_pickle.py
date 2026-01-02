"""
This class tests that you can properly pickle fit objects.

The old version used the reference datasets, but I think this isn't necessary;
it is totally reasonable just to use synthetic data, since this is cleaner
and requires less overhead (making this file easier to read).
"""
from numpy.testing import assert_equal

import os
import unittest
import powerlaw
import pickle

from collections.abc import Iterable

class TestPickling(unittest.TestCase):

    def test_pickle_dump(self):
        """
        Test that we can dump a Power_Law object.
        """
        dist = powerlaw.Power_Law(xmin=1, xmax=1e6, parameters=[1.5])

        with open('power_law_pickle_test.pkl', 'wb') as f:
            pickle.dump(file=f, obj=dist)

    def test_pickle_load(self):
        """
        Test that we can load the Power_Law pickle from ``test_pickle_dump``,
        and that its attributes are the same as the proper object
        """
        # If you change this, make sure to change the object from
        # test_pickle_dump
        dist1 = powerlaw.Power_Law(xmin=1, xmax=1e6, parameters=[1.5])

        with open('power_law_pickle_test.pkl', 'rb') as f:
            dist2 = pickle.load(file=f)

        # try block here so that we can make sure to delete the pickle file
        # after we are done, whether the test passes or fails.
        try:
            # __dict__ should give us just a list of properties excluding
            # functions (compared to dir(), which gives everything).

            # Dictionary comparison just makes sure that both dictionaries
            # have the same keys and values, though we have to use numpy's
            # assert_equal since some of the values (eg. data) might be
            # numpy arrays.
            assert_equal(dist1.__dict__, dist2.__dict__)

            # Also make sure we can call functions on the loaded object
            # The choice of generate_random is arbitrary.
            dist2.generate_random(10)

        finally:
            os.remove('power_law_pickle_test.pkl')

    def test_pickle_dump_load_fit(self):
        dist = powerlaw.Exponential(xmin=1, xmax=1e6, parameters=[1e-2])
        data = dist.generate_random(1000)

        fit1 = powerlaw.Fit(data, xmin=1, xmax=1e6)

        # Dump the object
        with open('fit_pickle_test.pkl', 'wb') as f:
            pickle.dump(file=f, obj=fit1)

        # try block here so that we can make sure to delete the pickle file
        # after we are done, whether the test passes or fails.
        try:
            # Now load the file
            with open('fit_pickle_test.pkl', 'rb') as f:
                fit2 = pickle.load(file=f)

            # Dictionary comparison just makes sure that both dictionaries
            # have the same keys and values, though we have to use numpy's
            # assert_equal since some of the values (eg. data) might be
            # numpy arrays.
            assert_equal(fit1.__dict__, fit2.__dict__)

            # Also make sure we can call functions on the loaded object
            # The choice of generate_random is arbitrary.
            fit2.exponential.generate_random(10)

        finally:
            os.remove('fit_pickle_test.pkl')


if __name__ == '__main__':
    # execute all TestCases in the module
    unittest.main()
