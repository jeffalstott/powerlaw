# Functions for definition and fitting of discretized log-normal distributions

# Assumes discretization-by-rounding, i.e. the probability that X=k is
# the probability that a continuous log-normal would lie between k-0.5 and
# k+0.5.

# The basic R system provides all the distributional functions for the
# continuous log-normal, which are used freely here.
# Cf. lnorm.R for "tail-conditional" functions

### Function for fitting:
# fit.lnorm.disc		Fit discretized log-normal to data, using
#				numerical likelihood maximization
### Backstage function, not intended for users:
# lnorm.tail.disc.loglike	Calculate log-likelihood
### Distribution functions (per R standards):
# dlnorm.disc			Probability mass function
# dlnorm.tail.disc		Tail-conditional probability mass function
# plnorm.tail.disc		Tail-conditional cumulative probability function


# Discretized probability mass function
# Input: Data vector, meanlog, sdlog, log parameter
# Output: Vector of (log) probabilities
dlnorm.disc <- function(x, meanlog, sdlog, log=FALSE) {
  # When x is very large, plnorm(x, lower.tail=TRUE) gets returned as 1,
  # but plnorm(x,lower.tail=FALSE) gets returned as a small but non-zero
  # number, so we should get fewer zeroes this way
  p <- plnorm(x-0.5,meanlog,sdlog,lower.tail=FALSE) - plnorm(x+0.5,meanlog,sdlog,lower.tail=FALSE)
  if (log) {
    return(log(p))
  } else {
    return(p)
  }
}

# Tail-conditional discretized probability mass function
# Returns NA if given values below threshold
# Input: Data vector, distribution parameters, lower threshold, log flag
# Output: Vector of (log) probabilities
dlnorm.tail.disc <- function(x,meanlog,sdlog,threshold=0,log=FALSE) {
  C <- plnorm(threshold-0.5,meanlog,sdlog,lower.tail=FALSE,log=log)
  if (log) {
   f <- function(y) {dlnorm.disc(y,meanlog,sdlog,log=TRUE) - C}
  } else {
   f <- function(y) {dlnorm.disc(y,meanlog,sdlog,log=FALSE)/C}
  }
  d <- ifelse(x<threshold,NA,f(x))
  return(d)
}

# Tail-conditional discretized cumulative probability function
# Returns NA if given values below threshold
# Input: Data vector, distributional parameters, lower threshold, usual flags
# Output: Vector of (log) probabilities
plnorm.tail.disc <- function(x,meanlog,sdlog,threshold=0,log.p=FALSE,
                             lower.tail=TRUE) {
  h <- function(y) { plnorm(y+0.5,meanlog,sdlog)/plnorm(threshold-0.5,meanlog,sdlog)}
  if (lower.tail) {
    g <- function(y) {1-h(y)}
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

lnorm.tail.disc.loglike <- function(x, meanlog, sdlog, threshold) {
  n <- length(x)
  JointProb <- sum(dlnorm.disc(x,meanlog,sdlog,log=TRUE))
  ProbOverThreshold <- plnorm(threshold-0.5,meanlog, sdlog, lower.tail=FALSE,
                              log.p=TRUE)
  return(JointProb - n*ProbOverThreshold)
}

# Fit a discretized log-normal to data, assumed integer-valued, by simple-minded
# likelihood maximization
fit.lnorm.disc <- function(x, threshold=0) {
  x.log <- log(x)
  theta_0 <- c(mean(x.log),sd(x.log))
  # Chop off values below threshold
  x <- x[x>=threshold]
  negloglike <- function(theta) {-lnorm.tail.disc.loglike(x,theta[1],theta[2],threshold)}
  est <- nlm(f=negloglike,p=theta_0)
  fit <- list(type="lnorm.disc",meanlog=est$estimate[1],sdlog=est$estimate[2],
              loglike=-est$minimum, threshold=threshold,
              datapoints.over.threshold = length(x))
  return(fit)
}
