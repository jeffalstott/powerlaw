R Code for The Estimation of Power Laws and Their Comparison to Heavy-Tailed
Alternatives


The contents of this directory should consist of some R functions for
estimating continuous and discrete power laws, and comparing them to various
more-or-less heavy-tailed alternative distributions.  It is inteded to
accompany the paper [1], and is likely to not be very comprehensible without
that paper.  This document gives a brief overview of the accompanying files and
their installation and usage.  It does NOT explain how to use R.  Some of the
files require the Gnu Scientific Library (GSL;
http://www.gnu.org/software/gsl); this document does not explain the GSL, or
how to compile against it.



CONTENTS OF THIS DIRECTORY

Power Law Distributions:
pareto.R			Definition and estimation of continuous power-
				law (Pareto) distributions
zeta.R				Definition and estimation of discrete power-law
				(zeta)
				distributions

Power Laws with Exponential Cut-offs:
powerexp.R			Continuous power law with exponential cut-off
discpowerexp.R			Discrete power law with exponential cut-off

Comparison to Alternatives:
power-law-test.R		Likelihood-based testing of power laws vs.
				alternatives

Alternative Distributions:
discexp.R			Discrete exponential distribution
disclnorm.R			Discretized log-normal distribution
discweib.R			Discrete stretched exponential (Weibull)
				distribution
exp.R				Exponential distribution (continuous)
lnorm.R				Log-normal distribution (continuous)
poisson.R			Poisson distribution (discrete)
weibull.R			Weibull (stretched exponential) distribution
yule.R				Yule (or "Yule-Simon") distribution

Ancillary C Code:
zeta-function.tgz		Hurwicz zeta function, for normalizing constant
				of zeta distribution; requires GSL.
exponential-integral.tgz	Exponential integral function, for normalizing
				constant of continuous power law with
				exponential cut-off; requires GSL.
discpowerexp.c			Normalizing constant for discrete power law with
				exponential cut-off.




OVERVIEW OF THE CODE AND ITS USAGE

The files pareto.R and zeta.R contain the functions which define the continuous
and discrete (respectively) power-law distributions, and other functions which
will estimate their scaling exponent in various ways.  (The method of
estimating the lower cut-off for the scaling region, described in [1], is
currently not yet implemented in R.)  The files for the alternative
distributions provide functions which will fit other distributions to the
right/upper tails of data sets.  Some of these distributions (e.g., the
log-normal) are already defined in R, in which case most of the code has to
do with estimation; in other cases (e.g. the Yule distribution), the code
also has to define the distribution.

The fitting functions --- at least the versions intended for users, generally
named things like "pareto.fit" or "lnorm.fit" --- return lists, with multiple
named components.  One component is always the type of distribution ("pareto",
"lnorm", etc.); other components give estimated parameters, and still others
give information either about the data (e.g., the number of samples) or the
fitting process itself (lower cut-off used for the tail, log likelihood, exact
fitting method, etc.).  See the code and comments for details.  I recommend
using the component names rather than their exact ordering in your code, for
comprehensibility and as a hedge against later changes.

The file power-law-test.R contains the functions for performing
likelihood-ratio tests for comparing the fit of power law distributions to
alternatives.  These functions ALL assume that the distributions are lists,
with the components given by the estimation functions.  (If you want to use
different estimation methods but the same testing code, therefore, you can.)

To test the hypothesis of a pure power law vs. a power law with an exponential
cut-off, do
	> power.powerexp.lrt(power.d,powerexp.d)
where "power.d" is the fitted power-law, and "powerexp.d" is the fitted power
law with exponential cut-off; this will return the actual log likelihood ratio,
and the appropriate (chi-squared) p-value.

To test the hypothesis of a power law against a non-nested alternative, it is
necessary to find the distribution of log-likelihood ratios over the data
set.  Different functions must be invoked, depending on the alternative.
For example,
	> vuong(pareto.lnorm.llr(x,x.pareto,x.lnorm))
will compare Pareto and log-normal distributions for the data set "x".  The
inner function, "pareto.lnorm.llr", calculates the distribution of
log-likelihood ratios for the Pareto (given by the argument "x.pareto") over
the log-normal (given by "x.lnorm").  Only points in "x" which equal or
exceed the lower cut-off embedded in "x.pareto" will be used.  The outer
function, "vuong", then implements Vuong's test of mis-specified non-nested
hypotheses.  It will give both a "one-sided" p-value, which is an upper limit
on getting that small a log likelihood ratio if the power law is actually true,
and a "two-sided" p-value, which is the probability of getting a log likelihood
ratio which deviates that much from zero in _either_ direction, if the two
distributions are actually equally good.  (See [1] for details.)  To perform
other comparisons, change the inner function, e.g., "pareto.exp.llr", or
"zeta.yule.llr".



INSTALLATION

Most of the files are straight-forward collections of R functions --- namely,
all of those named *.R.  These can be loaded into R by the "source" command,
e.g.,
	> source("pareto.R")
will load in "pareto.R", assuming the current R working directory is the
one where you've moved these files to.  This is all the installation necessary.

Three files, however, require special attention.  The discrete power law (zeta)
distribution, and both power laws with exponential cut-offs, involve nasty
numerical constants in their normalization.  These are calculated by separate C
programs, contained in the files "zeta-function.tgz",
"exponential-integral.tgz" and "discpowerexp.c".  The *.tgz files uncompress to
directories containing the C code program and makefiles; they require the Gnu
Scientific Library.  (See "zeta.R" and "powerexp.R" for more detailed
installation instructions.)  On the other hand, "discpowerexp.c" just needs to
be compiled.  (See "discpowerexp.R" for more details.)  All three files contain
constants telling R where to find the appropriate executable programs --- these
must be edited by the user, preferably before sourcing the R files.

COPYRIGHT

All code was written by Cosma Rohilla Shalizi (http://bactra.org/), 2004--2007,
and is copyright by him.  Use of the Gnu Scientific Library (GSL) is governed
by the terms of the appropriate accompanying Gnu Public License.  This code
itself is NOT released under the Gnu Public License, though future releases may
be; you are free to redistribute this code in its entirety, with this notice
attached.  If you use it in a scientific paper, please cite [1].  Bug reports
are gratefully received; technical support will not be provided.

Needless to say, this code comes with ABSOLUTELY NO WARRANTY.



ACKNOWLEDGMENTS

Thanks to Aaron Clauset, Christopher Genovese, Kristina Klinkner and Mark
Newman for valuable suggestions.



REFERENCES

[1] Aaron Clauset, Cosma Rohilla Shalizi, and M. E. J. Newman, "Power-law
Distributions in Empirical Data", http://arxiv.org/abs/0706.1062



REVISION/RELEASE HISTORY
v 0.0	2007-06-04	First release
v 0.0.1	2007-06-29	Fixed typo in pareto.R, compilation instructions in
			discpowerexp.R
v 0.0.2	2007-07-22	Fixed bug in plot.survival.loglog
v 0.0.3 2007-07-25	Fixed bug in discpowerexp.loglike
     


AGENDA

* Improve R-to-C interface, currently quite crude

* Implement sanity/type checking where appropriate (e.g. in testing code)

* Implement procedure to estimate the scaling threshold

* Add bootstrap calculation of severity levels to testing code

* Remove dependence on GSL by re-writing code for calculating the Hurwitz zeta
  function and the exponential integral function in my own C

* Turn this into a proper R package
