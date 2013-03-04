# Functions for estimation of an exponential distribution

# The use of an exponential to model values above a specified
# lower threshold can be done in one of two ways.  The first is simply to
# shift the standard exponential, i.e, to say that x-threshold ~ exp.
# The other is to say that Pr(X|X>threshold) = exp(X|X>threshold), i.e.,
# that the right tail follows the same functional form as the right tail of an
# exponential, without necessarily having the same probability of being in the
# tail.
# These will be called the "shift" and "tail" methods respectively.
# The shift method is, computationally, infinitely easier, but not so suitable
# for our purposes.

# The basic R system provides dexp (density), pexp (cumulative distribution),
# qexp (quantiles) and rexp (random variate generation)

### Functions for fitting:
# exp.fit		Fit exponential to data via likelihood maximization,
#			with choice of methods
### Backstage functions, not intended for users:
# exp.fit.tail		Fit exponential via "tail" method (default)
# exp.fit.moment	Fit exponential via "shift" method, starting with
#                       appropriate moments
# exp.loglike.shift	Calculate log likelihood of shifted exponential
# exp.loglike.tail	Calculate log likelihood of tail-conditional exponential

# Fit exponential distribution to data
# A wrapper for actual methods, defaulting to the "tail" method
exp.fit <- function(x,threshold=0,method="tail") {
  switch(method,
    tail = { fit <- exp.fit.tail(x,threshold) },
    shift = { fit <- exp.fit.shift(x,threshold) },
    {
       cat("Unknown method in exp.fit\n")
       fit <- NA}
  )
  return(fit)
}

exp.fit.tail <- function(x,threshold = 0) {
  # Start with a global estimate of the parameter
  lambda_0 <- exp.fit.moment(x,method="tail")$rate
  x <- x[x>=threshold]
  # The function just called ignores values of method other than "shift"
  # but let's not take chances!
  negloglike <- function(lambda) { -exp.loglike.tail(x,lambda,threshold) }
  fit <-nlm(f=negloglike,p=lambda_0)
  list(type="exp", rate=fit$estimate, loglike=-fit$minimum, datapoints.over.threshold=length(x))
}

exp.fit.moment <- function(x, threshold = 0, method="shift") {
  x <- x[x>=threshold]
  if (method=="shift") { x <- x-threshold }
  lambda <- 1/mean(x)
  loglike <- exp.loglike.shift(x, lambda, threshold)
  list(type="exp", rate=lambda, loglike=loglike, datapoints.over.threshold=length(x))
}

exp.loglike.shift <- function(x, lambda, threshold=0) {
  # Assumes (X-threshold) is exponentially distributed
  # See Johnson and Kotz, ch. 18 for more on this form of the distribution
  x <- x[x>=threshold]
  x <- x-threshold
  sum(dexp(x,rate=lambda,log=TRUE))
}

exp.loglike.tail <- function(x, lambda, threshold=0) {
  # We want p(X=x|X>=threshold) = p(X=x)/Pr(X>=threshold)
  x <- x[x>=threshold]
  n <- length(x)
  Like <- exp.loglike.shift(x,lambda)
  ThresholdProb <- pexp(threshold, rate=lambda, log=TRUE, lower.tail=FALSE)
  Like - n*ThresholdProb
}
