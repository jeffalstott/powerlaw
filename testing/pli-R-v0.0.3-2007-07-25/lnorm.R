# Functions for estimation of lognormal distributions

# The use of a log-normal distribution to model values above a specified
# lower threshold can be done in one of two ways.  The first is simply to
# shift the standard log-normal, i.e, to say that x-threshold ~ lnorm.
# The other is to say that Pr(X|X>threshold) = lnorm(X|X>threshold), i.e.,
# that the right tail follows the same functional form as the right tail of a
# lognormal, without necessarily having the same probability of being in the
# tail.
# These will be called the "shift" and "tail" methods respectively.
# The shift method is, computationally, infinitely easier, but not so suitable
# for our purposes.

# The basic R system provides dlnorm (density), plnorm (cumulative
# distribution), qlnorm (quantiles) and rlnorm (random variate generation)

### Function for fitting:
# lnorm.fit		Fit log-normal to data, with choice of methods
### Back-stage functions not intended for users:
# lnorm.fit.max		Fit log-normal by maximizing likelihood ("tail")
# lnorm.fit.moments	Fit log-normal by moments of log-data ("shift")
# lnorm.loglike.tail	Tail-conditional log-likelihood of log-normal
# lnorm.loglike.shift	Log-likelihood of shifted log-normal
### Tail distribution functions (per R standards):
# dlnorm.tail		Tail-conditional probability density
# plnorm.tail		Tail-conditional cumulative probability


# Fit log-normal to data over threshold
# Wrapper for the shift (log-moment) or tail-conditional (maximizer) functions
# Input: Data vector, lower threshold, method flag
# Output: List giving distribution type ("lnorm"), parameters, log-likelihood
lnorm.fit <- function(x,threshold=0,method="tail") {
  switch(method,
   tail = {
    if (threshold>0) { fit <- lnorm.fit.max(x,threshold) }
    else { fit <- lnorm.fit.moments(x) }
  },
  shift = {
    fit <- lnorm.fit.moments(x,threshold)
  },
  {
    cat("Unknown method\n")
    fit <- NA
  })
  return(fit)
}

# Fit log-normal by direct maximization of tail-conditional log-likelihood
# Note that direct maximization of the shifted lnorm IS lnorm.fit.moments, so
# that should be used instead for numerical-accuracy reasons
# Input: Data vector, lower threshold
# Output: List giving distribution type ("lnorm"), parameters, log-likelihood
lnorm.fit.max <- function(x, threshold = 0) {
  # Use moment-based estimator on whole data as starting point
  initial.fit <- lnorm.fit.moments(x)
  theta_0 <- c(initial.fit$meanlog,initial.fit$sdlog)
  x <- x[x>=threshold]
  negloglike <- function(theta) {
    -lnorm.loglike.tail(x,theta[1],theta[2],threshold)
  }
  est <- nlm(f=negloglike,p=theta_0)
  fit <- list(type="lnorm",meanlog=est$estimate[1],sdlog=est$estimate[2],
              datapoints.over.threshold = length(x), loglike=-est$minimum)
  return(fit)
}

# Fit log-normal via moments of the log data
# This is the maximum likelihood solution for the shifted lnorm
# Input: Data vector, lower threshold
# Output: List giving distribution type ("lnorm"), parameters, log-likelihood
lnorm.fit.moments <- function(x, threshold = 0) {
  x <- x[x>=threshold]
  x <- x-threshold
  LogData <- log(x)
  M = mean(LogData)
  SD = sd(LogData)
  Lambda = lnorm.loglike.shift(x,M,SD,threshold)
  fit <- list(type="lnorm", meanlog=M, sdlog=SD, loglike=Lambda)
  return(fit)
}

# Tail-conditional log-likelihood of log-normal
# Input: Data vector, distributional parameters, lower threshold
# Output: Real-valued log-likelihood
lnorm.loglike.tail <- function(x, mlog, sdlog, threshold = 0) {
  # Compute log likelihood of data under assumption that the generating
  # distribution is a log-normal with the given parameters, and that we
  # are only looking at the tail values, x >= threshold
  # We want p(X=x|X>=threshold) = p(X=x)/Pr(X>=threshold)
  x <- x[x>= threshold]
  n <- length(x)
  Like <- lnorm.loglike.shift(x,mlog, sdlog)
  ThresholdProb <- plnorm(threshold, mlog, sdlog, log.p=TRUE, lower.tail=FALSE)
  L <- Like - n*ThresholdProb
  return(L)
}

# Loglikelihood of shifted log-normal
# Input: Data vector, distributional parameters, lower threshold
# Output: Real-valued log-likelihood
lnorm.loglike.shift <- function(x, mlog, sdlog, x0=0) {
  # Compute log likelihood under assumption that x-x0 is lognromally
  # distributed
  # This (see Johnson and Kotz) the usual way of combining a lognormal
  # distribution with a hard minimum value.  (They call the lower value theta)
  x <- x[x>=x0]
  x <- x-x0
  L <- sum(dlnorm(x,mlog,sdlog,log=TRUE))
  return(L)
}


# Tail-conditional density function
# Returns NA if given values below the threshold
# Input: Data vector, distributional parameters, lower threshold, log flag
# Output: Vector of (log) probability densities
dlnorm.tail <- function(x, meanlog, sdlog, threshold=0,log=FALSE) {
  # Returns NAs at positions where the values in the input are < threshold
  if (log) {  
    f <- function(x) {dlnorm(x,meanlog,sdlog,log=TRUE) - plnorm(threshold,meanlog,sdlog,log=TRUE)}
  } else {
    f <- function(x) {dlnorm(x,meanlog,sdlog)/plnorm(threshold,meanlog,sdlog)}
  }
  d <- ifelse(x<threshold,NA,f(x))
  return(d)
}

# Tail-conditional cumulative distribution function
# Returns NA if given values below the threshold
# Input: Data vector, distributional parameters, lower threshold, usual flags
# Output: Vector of (log) probabilities
plnorm.tail <- function(x,meanlog,sdlog,threshold=0,lower.tail=TRUE,log.p=FALSE) {
  c <- plnorm(threshold,meanlog,sdlog,lower.tail=FALSE)
  c.log <- plnorm(threshold,meanlog,sdlog,lower.tail=FALSE,log.p=TRUE)
  if ((!lower.tail) && (!log.p)) {
    f <- function(x) {plnorm(x,meanlog,sdlog,lower.tail=FALSE)/c}
  }
  if ((lower.tail) && (!log.p)) {
    f <- function(x) {1 - plnorm(x,meanlog,sdlog,lower.tail=FALSE)/c}
  }
  if ((!lower.tail) && (log.p)) {
    f <- function(x) {plnorm(x,meanlog,sdlog,lower.tail=FALSE,log.p=TRUE) - c.log}
  }
  if ((lower.tail) && (log.p)) {
    f <- function(x) {log(1 - plnorm(x,meanlog,sdlog,lower.tail=FALSE)/c)}
  }
  p <- ifelse(x<threshold,NA,f(x))
  return(p)
}

