# -*- coding: utf-8 -*-

import unittest
import numpy as np

import walkobj as wo

import walkobj as wo
import numpy as np


class MyClass:
    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar


c1 = MyClass('foo', np.array([1, 2, 3]))
c2 = MyClass('foo', np.array([1, 2, 3]))
c3 = MyClass('food', np.array([1, 2, 3]))
c4 = MyClass('foo', np.array([1, 2, 3, 4]))

testvals = [("abc", "abcd", False),
            (1, 1.0, True),
            (1, 1.01, False),
            (True, False, False),
            (True, True, True),
            (True, 1, True),
            (True, 0, False),
            (True, "abc", False),
            (False, "abc", False),
            (None, "abc", False),
            (None, 1, False),
            (None, 0, False),
            (np.nan, True, False),
            (np.nan, False, False),
            (np.nan, 0, False),
            (np.nan, np.nan, False),
            ([1, 2, 3], [1, 2, 3], True),
            ([1, 2, 3], [1, 2, 3, 4], False),
            ([1, 2, 3], [1, 2], False),
            ({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}, True),
            ({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3.01}, False),
            ({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2}, False),
            ([c1, [c1, c2]], [c2, [c2, c1]], True),
            ((c1, (c1, c2)), (c2, (c2, c1)), True),
            ((c1, c2), (c2, c1), True),
            (c1, c3, False),
            ((c1, (c1, (c1, c4))), (c1, (c1, (c1, c3))), False)]


def test_deep_equals(obja, objb, isequal):
    """Confirm that equality of two objects is correct

    obja    : One object
    objb    : Second object
    isequal : Whether or not they should be equal

    Converts them to typedtrees and uses ==.

    Returns True iff they have the right equality.

    Note - This doesn't work for NaNs, as two NaNs are not considered equal
    """

    objatree = wo.typedtree(obja)
    objbtree = wo.typedtree(objb)
    match = objatree == objbtree
    ok = match == isequal

    if ok:
        s = "pass"
    else:
        s = "fail"

    print(f"{obja} == {objb} is {match} : {s}")
    return ok


class FirstTestCase(unittest.TestCase):

    def test_power_pickle_dump(self):
        print("Testing walkobj.typedtree equality for different objects")

        for args in testvals:
            assert test_deep_equals(*args) is True


if __name__ == '__main__':
    # execute all TestCases in the module
    unittest.main()
