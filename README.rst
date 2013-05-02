`powerlaw` is a toolbox using the statistical methods developed in
`Clauset et al. 2007`__ and `Klaus et al. 2011`__ to determine if a
probability distribution fits a power law. This package is in "open beta",
which means everything pretty much works but it's being tweaked and expanded
on. Academics, please cite as:

Jeff Alstott, Ed Bullmore, Dietmar Plenz. (2013). powerlaw: a Python package
for analysis of heavy-tailed distributions. `arXiv:1305.0215 [physics.data-an]`__

__ http://arxiv.org/abs/0706.1062 
__ http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0019779
__ http://arxiv.org/abs/1305.0215

Basic Usage 
-----------------
For the simplest, typical use cases, this tells you everything you need to
know.::

    import powerlaw
    data = array([1.7, 3.2 ...]) #data can be list or Numpy array
    results = powerlaw.Fit(data)
    print results.power_law.alpha
    print results.power_law.xmin
    R, p = results.distribution_compare('power_law', 'lognormal')

For more explanation, understanding, and figures, see the working paper,
which illustrates all of powerlaw's features. For details of the math, see
Clauset et al. 2007, which developed these methods.

Quick Links
-----------------
`Installation`__

`Working paper illustrating all of powerlaw's features, with figures`__

`Code examples from manuscript, as an IPython Notebook`__

`Documentation`__

`Known Issues`__

`Update Notifications, Mailing List, and Contacts`__

Note! This code works on Python 2.x, not 3.x.
This code was developed and tested with the `Enthought Python Distribution`__, 
and will update to 3.x whenever Enthought updates to 3.x.
The full version of Enthought is `available for free for academic use`__.

__ http://code.google.com/p/powerlaw/wiki/Installation
__ http://arxiv.org/abs/1305.0215 
__ http://nbviewer.ipython.org/19fcdd6a4ba400ce8de2
__ http://pythonhosted.org/powerlaw/
__ https://code.google.com/p/powerlaw/wiki/KnownIssues
__ http://code.google.com/p/powerlaw/wiki/Interact
__ http://www.enthought.com/products/epd.php
__ http://www.enthought.com/products/edudownload.php 

Acknowledgements
-----------------
Many thanks to  Andreas Klaus, Mika Rubinov and Shan Yu for helpful
discussions. Thanks also to `Andreas Klaus <http://neuroscience.nih.gov/Fellows/Fellow.asp?People_ID=2709>`_,
`Aaron Clauset, Cosma Shaliz <http://tuvalu.santafe.edu/~aaronc/powerlaws/>`_,
and `Adam Ginsburg <http://code.google.com/p/agpy/wiki/PowerLaw>`_ for making 
their code available. Their implementations were a critical starting point for
making powerlaw.
