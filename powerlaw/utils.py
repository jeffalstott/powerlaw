"""
Utilities used for managing data.
"""
import numpy as np

def checkunique(data):
    """Quickly checks if a sorted array is all unique elements."""
    for i in range(len(data)-1):
        if data[i]==data[i+1]:
            return False
    return True

#def checksort(data):
#    """
#    Checks if the data is sorted, in O(n) time. If it isn't sorted, it then
#    sorts it in O(nlogn) time. Expectation is that the data will typically
#    be sorted. Presently slower than numpy's sort, even on large arrays, and
#    so is useless.
#    """
#
#    n = len(data)
#    from numpy import arange
#    if not all(data[i] <= data[i+1] for i in arange(n-1)):
#        from numpy import sort
#        data = sort(data)
#    return data

def bisect_map(mn, mx, function, target):
    """
    Uses binary search to find the target solution to a function, searching in
    a given ordered sequence of integer values.

    Parameters
    ----------
    seq : list or array, monotonically increasing integers
    function : a function that takes a single integer input, which monotonically
        decreases over the range of seq.
    target : the target value of the function

    Returns
    -------
    value : the input value that yields the target solution. If there is no
    exact solution in the input sequence, finds the nearest value k such that
    function(k) <= target < function(k+1). This is similar to the behavior of
    bisect_left in the bisect package. If even the first, leftmost value of seq
    does not satisfy this condition, -1 is returned.
    """
    if function([mn]) < target or function([mx]) > target:
        return -1
    while 1:
        if mx==mn+1:
            return mn
        m = (mn + mx) / 2
        value = function([m])[0]
        if value > target:
            mn = m
        elif value < target:
            mx = m
        else:
            return m

