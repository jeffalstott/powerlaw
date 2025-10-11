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

def bisect_map(mn, mx, function, target, tol):
    """
    Uses binary search to find the target solution to a function, searching in
    a given ordered sequence of integer values.

    Parameters
    ----------
    mn : int or float
        The lower bound starting point.

    mx : int or float
        The upper bound starting point.

    function : callable
        A functioe that takes a single value as an input, and returns
        as a single value as output. Can either monotonically
        increase or decrease over the range ``[mn, mx]``.

    target : int or float
        The target value of the function.

    tol : int or float
        The tolerance for stopping the search: when the midpoint of the
        two bounds changes by less than this value, the search will end.

    Returns
    -------
    value : the input value that yields the target solution. If there is no
    exact solution in the input sequence, finds the nearest value k such that
    function(k) <= target < function(k+1). This is similar to the behavior of
    bisect_left in the bisect package. If even the first, leftmost value of seq
    does not satisfy this condition, -1 is returned.
    """
    # Determine if our function is monotonically increasing or decreasing
    increasing_func = function(mx) > function(mn)

    # Make sure that our target is within the range of mx and mn
    if (increasing_func and function(mx) > target and function(mn) < target) or \
       (not increasing_func and function(mn) > target and function(mx) < target):

        old_m = 0
        m = 0
        while True:
            # Take the midpoint
            old_m = m
            m = (mn + mx) / 2

            # If the difference between the old and new values is less than the prescribed
            # tolerance, we exit
            if np.abs(old_m - m) < tol:
                return m

            value = function(m)

            # Divide the region in half
            # Depends on whether we have an increasing or decreasing
            # function

            # For an increasing function where the current value is
            # greater than the target we should move the upper bound down
            if value > target and increasing_func:
                mx = m

            # For a decreasing function where the current value is
            # greater than the target we should move the lower bound up
            elif value > target and not increasing_func:
                mn = m

            elif value < target and increasing_func:
                mn = m

            elif value < target and not increasing_func:
                mx = m

            else:
                return m

    else:
        return None
