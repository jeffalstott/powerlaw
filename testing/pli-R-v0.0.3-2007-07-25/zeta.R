# Functions for the Zipf and generalized Zipf distribution, a.k.a. the
# zeta function distribution

# The distribution is defined by
# Pr(X = k) \propto k^-s
# for s >= q.  The constant of proportionality is the Hurwitz zeta
# function,
# zeta(s,q) = \sum_{k=q}^{\infty}{k^{-s}}
#           = \sum_{k=0}^{\infty}{(k+q)^{-s}}
# the latter being the form given by the Gnu Scientific Library (GSL).  The
# Riemann zeta function is the special case q = 1.

# The normalizing constant is evaluated by a (comparatively crude) call
# to the GSL, embodied in a stand-alone piece of C.  This must be compiled
# and put someplace R can execute it.  It can be found in a file called
# zeta-function.tgz, which should accompanying this code.
# To install it on a Unix system, first make sure GSL is installed.
### > tar xzf zeta-function.tgz
### > cd zeta-function
# At this point, you need to edit the file "Makefile" to give the location of
# the GSL, which on my system is /sw/lib (for the library) and /sw/include
# (for the included files).  Then
### > make
### > mv zeta_func yourexecutablepath
# where "yourexecutablepath" is a directory where you can put executable
# files.  Then edit the variable "zeta_function_filename" in this file,
# below, to give the full path to the executable program zeta_func.
# This is not the world's slickest installation mechanism, no.


### Function for fitting to data:
# zeta.fit		Fit to tail of data, with choice of methods
### Distributional functions, per R standards:
# dzeta			Probability mass function
# pzeta			Cumulative probability function
### Backstage functions, not for users:
# zeta.fit.direct	Fit by numerical maximization of likelihood (default)
# zeta.fit.approx	Fit using Mark Newman's analytical approximation
# zeta.loglike		Calculate log-likelihood value
# zeta_func		Calculate zeta function for multiple thresholds
# zeta_func_once.gsl	Calculate zeta function for one point by an external
#			call to the GSL


# The location of the external program calculating the zeta function
# Used by zeta_function_once.gsl
zeta_function_filename <- "~/bin/zeta_function"
### EDIT THIS LOCATION!!! ####




# Density (probability mass) of discrete power law
# Returns NA on values < threshold
# Input: Data vector, lower threshold, scaling exponent, log flag
# Output: Vector of (log) densities (probabilities)
dzeta <- function(x, threshold=1, exponent, log=FALSE) {
  C <- zeta_func(exponent,threshold)
  if (log) {
    f <- function(y) {-log(C) - exponent*log(y)}
  } else {
    f <- function(y) {(1/C) * (y^(-exponent))}
  }
  d <- ifelse(x<threshold,NA,f(x))
  return(d)
}

# Cumulative probability of discrete power law
# Returns NA on values < threshold
# Input: Data vector, lower threshold, scaling exponent, usual flags
# Output: Vector of (log) probabilties
pzeta <- function(x, threshold=1, exponent, lower.tail=TRUE, log.p=FALSE) {
  C <- zeta_func(exponent,threshold)
  h <- function(y) { zeta_func(exponent,y)/C }
  if (lower.tail) {
    g <- function(y) { 1-h(y) }
  } else {
    g <- function(y) {h(y)}
  }
  if (log.p) {
    f <- function(y) {log(g(y))}
  } else {
    f <- function(y) {g(y)}
  }
  p <- ifelse(x<threshold,NA,f(x))
  return(p)
}

# Calculate tail-conditional log-likelihood
# Will give an ERROR if any of the data values < threshold
# Input: Data vector, threshold, exponent
# Output: Log likelihood
zeta.loglike <- function(x,threshold=1,exponent) {
  L <- sum(dzeta(x,threshold,exponent,log=TRUE))
  return(L)
}

# Fit data (assumed discrete) to a discrete power law via maximum likelihood
# Wrapper for two subsidiary methods, direct optimization, and applying
# an approximate formula due to M. E. J. Newman
# Input: Data vector, lower cut-off, method flag
# Output: List giving distribution type, estimated parameter, information
#	  on fitting and data
zeta.fit <- function(x, threshold = 1, method="ml.direct") {
  x <- x[x>=threshold]
  n <- length(x)
  switch(method,
    ml.direct = {alpha <- zeta.fit.direct(x,threshold)},
    ml.approx = {alpha <- zeta.fit.approx(x,threshold)},
    {cat("Unknown method in zeta.fit",method,"\n"); return(NA)}
 )
 loglike <- zeta.loglike(x,threshold,alpha)
 fit <- list(type="zeta", threshold=threshold, exponent=alpha,
             loglike=loglike, samples.over.threshold=n, method=method)
 return(fit)
}

# Fit data (assumed discrete) to a discrete power law via numerical optimization
# of the likelihood
# "Primes" the optimization by using the approximate formula
# This function should not be called directly by users, but indirectly via
# zeta.fit, which will do pre-processing and sanity-checking
# Input: Data vector, lower cut-off
# Output: Estimated scaling exponent
zeta.fit.direct <- function(x, threshold) {
  # Use approximate method to start optimization
  alpha_0 <- zeta.fit.approx(x,threshold)
  negloglike <- function(alpha) {-zeta.loglike(x,threshold,alpha)}
  est <- nlm(f=negloglike,p=alpha_0)
  # Extract and return the parameter
  return(est$estimate[1])
}

# Fit data (assumed discrete) to a discret power law via Mark Newman's
# approximate formula
# Slightly duplicates code in pareto.R --- commented-out line in fact directly
# calls pareto.R
# The approximation appears to work quite well when threshold >= 20, and can
# even be reasonable when >= 3.
# This function should not be called directly by users, but indirectly via
# zeta.fit, which will do pre-processing and sanity-checking
# Input: Data vector, lower cut-off
# Output: Estimated scaling exponent
zeta.fit.approx <- function(x,threshold) {
  effective.threshold <- threshold - 0.5
  # alpha <- pareto.fit(x,effective.threshold,method="ml")$exponent
  n <- length(x)
  sum.of.logs <- sum(log(x))
  xmin.factor <- n*log(threshold-0.5)
  alpha <- 1 + n/(sum.of.logs - xmin.factor)
  return(alpha)
}
  

# Compute the Hurwitz zeta function multiple times, with multiple additive
# constants (lower limits of summation)
# First argument is the exponent, second is the limit of summation
# Works by invoking the zeta_func_once.gsl function
# Input: Exponent, vector of thresholds
# Output: Vector of Hurwitz zeta values
zeta_func <- function(s, q) {
  zetas <- sapply(q, zeta_func_once.gsl, s)
  return(zetas)
}

# Compute the Hurwitz zeta via invoking the GNU scientific library ONCE
# Input: real-valued exponent (s), additive constant (q)
# Output: Real value
zeta_func_once.gsl <- function(q,s) {
  zeta.command <- paste(zeta_function_filename,s,q)
  zeta <- as.numeric(system(zeta.command,intern=TRUE))
  return(zeta)
}
