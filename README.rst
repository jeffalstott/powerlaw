`powerlaw` is a toolbox using the statistical methods developed in
`Clauset et al. 2007`__ and `Klaus et al. 2011`__ to determine if a
probability distribution fits a power law. This package is in "open beta",
which means everything pretty much works but it's being tweaked and expanded
on. Academics, please cite as:

    Jeff Alstott. (2012). powerlaw Python package. Web address:
    pypi.python.org/pypi/powerlaw.


__ http://arxiv.org/abs/0706.1062 
__ http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0019779

Basic Usage 
-----------------
For the simplest, typical use cases, that tells you everything you need to
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

`Known Issues`__

`Update Notifications, Mailing List, and Contacts`__

`Note! This code works on Python 2.x, not 3.x.
This code was developed and tested with the `Enthought Python Distribution`__
 and will update to 3.x whenever Enthought updates to 3.x.
The full version of Enthought is `available for free for academic use`__.

__ http://code.google.com/p/powerlaw/wiki/Installation
__ https://powerlaw.googlecode.com/files/powerlaw.pdf
__ https://code.google.com/p/powerlaw/wiki/KnownIssues
__ http://code.google.com/p/powerlaw/wiki/Interact
__ http://www.enthought.com/products/epd.php
__ http://www.enthought.com/products/edudownload.php 

Acknowledgements
-----------------
Many thanks to Mika Rubinov and Shan Yu for helpful discussions and to Adam
Ginsburg for posting `his code`__, which inspired the xmin-selection function
of this toolbox.

__ http://code.google.com/p/agpy/wiki/PowerLaw
