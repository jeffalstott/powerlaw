Saving and loading fits
==================================

After choosing parameters, calculating ``xmin`` values, or fitting various
distributions, you might want to save the ``Fit`` object to a file. This
allows you to easily load it in during a future session, without having
to recalculate everything (particularly the ``xmin`` value, which is usually
somewhat computationally expensive).

This can be done easily using :meth:`Fit.save` and :meth:`Fit.load`.

.. code-block::

    data = [1.1, 5.3, 3.7, ...]
    fit = powerlaw.Fit(data, xmin=0.1)

    fit.save('output.h5')

.. code-block::

    # In another session

    fit = powerlaw.Fit.load('output.h5')

    fit.plot_pdf()
    ...

The saving and loading functions currently support two different file formats:
`pickle <https://docs.python.org/3/library/pickle.html>`_ and
`hdf5 <https://www.hdfgroup.org/solutions/hdf5/>`_.
A pickle file is Python's way of serializing an object,
which saves the entire object to a file that can then be loaded later. The
hdf5 format is a more universal format that allows you to save numerical data
alongside various metadata. This means that the hdf5 file doesn't contain the
actual ``Fit`` object like the pickle file does, but instead saves all of
the important information and then reconstructs the ``Fit`` when you load it
back in. Pickling is done using the `dill <https://github.com/uqfoundation/dill>`_
library (which improves on the standard library ``pickle``) and hdf5 file
operations are done using the `h5py <https://github.com/h5py/h5py>`_ library.

You can choose which format to use by either including it in the filename,
or with the ``format`` keyword:

.. code-block::

    fit.save('output.h5') # saves in hdf5 format
    fit.save('output', format='h5') # saves in hdf5 format
    fit.save('output.pkl') # saves in pickle format
    fit.save('output', format='pkl') # saves in pickle format

If you're just working with the ``powerlaw`` library, these two formats are
almost entirely interchangeable, with hdf5 files being slightly smaller than
pickle files. That being said, hdf5 files do have the advantage of being
easily read and interpreted outside of this library, or even outside of
Python altogether. If you're worried about future-proofing your data, or
want to use this data in other programming languages, hdf5 is probably better.


Automatic caching
-----------------

``powerlaw`` offers the option automatically cache *all* fits, if you don't
want to have to manually save files. This is disabled by default, but can
be enabled by setting the cache directory with :meth:`powerlaw.Fit.set_cache_folder()`.

.. code-block::

    powerlaw.Fit.set_cache_folder('data/')

For the rest of the session, all ``Fit`` objects will automatically be
saved in this folder after creation. And if you create a ``Fit`` object that
is identical to a cached one, it will be loaded instead of recalculating things.
This might be useful if you are working on a project where you are consistently
working with several predefined datasets, and you don't want to have to, for
example, recalculate ``xmin`` during each session.

.. code-block::

    powerlaw.Fit.set_cache_folder('data/')

    data = np.genfromtxt('data.txt')

    # This will calculate xmin, and then cache the object
    fit = powerlaw.Fit(data)

.. code-block::

    # In another session

    powerlaw.Fit.set_cache_folder('data/')

    # The same data as before
    data = np.genfromtxt('data.txt')

    # This will just load the previously cached file
    fit = powerlaw.Fit(data)

This replacement only happens when the data and all of the parameters of 
fitting are exactly the same.

.. code-block::

    # In another session

    # The same data as before
    data = np.genfromtxt('data.txt')

    # This will *not* load the previously cached file since xmin is different
    fit = powerlaw.Fit(data, xmin=1)


A note on constraints
---------------------

Constraint functions are a little tricky to save since they might have
dependencies on variables, functions or libraries beyond the function itself.
For example, the following constraint could very likely give an error:

.. code-block::

    import powerlaw
    import numpy as np

    data = np.genfromtxt('data.txt')

    def constraint(dist):
        """
        Some constraint that depends on the library numpy
        """
        E = np.exp(...)
        ...

    constraint_dict = {"type": 'eq',
                       "fun": constraint}

    fit = powerlaw.Fit(data, parameter_constraints=constraint_dict)

    fit.save('output.h5')

.. code-block::

    # In another session

    import powerlaw
    # numpy is *not* imported

    fit = powerlaw.Fit.load('output.h5')

    constraint = fit.parameter_constraints[0]["fun"]

    # This will give an error that the function can't find numpy since we
    # haven't imported it.
    constraint(...)

The best practice here is to have constraint functions be fully self contained,
including definitions of variables and library imports.

.. code-block::

    # Best practice: fully self-contained
    def constraint(dist):
        import numpy as np

        T = 100
        E = np.exp(-dist.Lambda * T)
        ...

.. code-block::

    # Not good practice but will still work
    T = 100
    def constraint(dist):
        import numpy as np

        E = np.exp(-dist.Lambda * T)
        ...

.. code-block::

    # Will not work!
    import numpy as np
    T = 100
    def constraint(dist):

        E = np.exp(-dist.Lambda * T)
        ...
