"""walkobj - object walker module

Functions for walking the tree of an object

Usage:
   Recursively print values of slots in a class instance obj:
      walk(obj, print)
   Recursively print values of all slots in a list of classes:
      walk([obj1, obj2, ...], print)
   Convert an object into a typed tree
      typedtree(obj)

A typed tree is a (recursive) list of the items in an object, prefixed by the type of the object
"""

from collections.abc import Iterable


BASE_TYPES = [str, int, float, bool, type(None)]


def atom(obj):
    """Return true if object is a known atomic type.

    obj : Object to identify

    Known atomic types are strings, ints, floats, bools & type(None).
    """
    return type(obj) in BASE_TYPES


def walk(obj, fun):
    """Recursively walk object tree, applying a function to each leaf.

    obj : Object to walk
    fun : function to apply

    A leaf is an atom (str, int, float, bool or None), or something
    that is not a dictionary, a class instance or an iterable object.
    """
    if atom(obj):
        fun(obj)
    elif isinstance(obj, dict):
        for keyobj in obj.items():
            walk(keyobj, fun)
    elif isinstance(obj, Iterable):
        for item in obj:
            walk(item, fun)
    elif '__dict__' in dir(obj):
        walk(obj.__dict__, fun)
    else:
        fun(obj)


def typedtree(obj):
    """Convert object to a typed nested list.

    obj : Object to walk

    Returns the object if it's an atom or unrecognized.

    If it's a class, return the list [type(obj)] + nested list of the
    object's slots (i.e., typedtree(obj.__dict__).

    If it's iterable, return the list [type(obj)] + nested list of
    items in list.

    Otherwise, it's unrecognized and just returned as if it's an atom.
    """
    if atom(obj):
        return obj
    if isinstance(obj, dict):
        return (dict, [typedtree(obj) for obj in obj.items()])
    if isinstance(obj, Iterable):
        return (type(obj), [typedtree(obj) for obj in obj])
    if '__dict__' in dir(obj):
        return (type(obj), typedtree(obj.__dict__))
    return obj
