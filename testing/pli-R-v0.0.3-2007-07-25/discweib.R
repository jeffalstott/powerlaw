# Functions for definition and fitting of discrete Weibull distributions

# There are several functions which all claim to be the "discrete Weibull"
# distribution, this code uses the Nakagawa-Osaki discretization, see
# http://ljk.imag.fr/membres/Olivier.Gaudoin/ICRSA03Gaudoin.pdf for
# lists of others

# Uses functions embedded in weibull.R, as well as R's built-in functions
# for the Weibull distribution --- run
### source("weibull.R")
# first --- you may need to modify the path to make sure R can find the right
# file!

### Function for fitting:
# discweib.fit		Fit discrete Weibull to tail of data via numerical
#			likelihood maximization
### Distributional functions, per R standards:
# ddiscweib		Probability mass function
# pdiscweib		Cumulative probability function
### Backstage function, not intended for users:
# discweib.loglike	Calculate log-likelihood


# Probability mass function for discrete Weibull distribution, conditional on
# being in the right/upper tail
# Input: Data vector, distributional parameters, lower cut-off, log flag
# Output: Vector of (log) probabilities
ddiscweib <- function(x,shape,scale,threshold=0,log=FALSE) {
  # Compute the PMF as the increments of the distribution function
  f <- function(y) { 
    p1 <- pdiscweib(y,shape,scale,threshold,lower.tail=FALSE)
    p2 <- pdiscweib(y+1,shape,scale,threshold,lower.tail=FALSE)
    return(p1-p2)
  }
  f.log <- function(y) {
    # Do calculations on a logarithmic scale
    # Let log(b) - log(a) = h = log(b/a), b > a
    # Then b-a = a(b/a - 1)
    # b-a = a(exp(log(b/a)) - 1)
    # b-a = a(exp(h) - 1)
    # log(b-a) = log(a) + log(exp(h) - 1)
    lp1 <- pdiscweib(y,shape,scale,threshold,lower.tail=FALSE,log.p=TRUE)
    lp2 <- pdiscweib(y+1,shape,scale,threshold,lower.tail=FALSE,log.p=TRUE)
    return(lp2 + log(exp(lp1-lp2)-1))
  }
  if(log) {
    d <- ifelse(x<threshold,NA,f.log(x))
  } else {
    d <- ifelse(x<threshold,NA,f(x))
  }
  return(d)
}


# Cumulative distribution function for discrete Weibull, conditional on being
# in the right/uppper tail
# Input: Data vector, distributional parameters, lower cut-off, usual flags
# Output: Vector of (log) probabilities
pdiscweib <- function(x,shape,scale,threshold=0,lower.tail=TRUE,log.p=FALSE) {
  # g(y) here is the probability of being strictly > y
  if (threshold == 0) {
    C <- 1
  } else {
    C <- pweibull(threshold,shape,scale,lower.tail=FALSE)
  }
  C.log <- log(C)
  g <- function(y) {pweibull(y,shape,scale,lower.tail=FALSE)/C}
  g.log <- function(y) { pweibull(y,shape,scale,lower.tail=FALSE,log.p=TRUE)-C.log}
  if (!lower.tail && !log.p) {
    f <- function(y) {g(y)}
  }
  if (!lower.tail && log.p) {
    f <- function(y) {g.log(y)}
  }
  if (lower.tail && !log.p) {
    f <- function(y) {1-g(y)}
  }
  if (lower.tail && log.p) {
    f <- function(y) {log(1-g(y))}
  }
  p <- ifelse(x<threshold,NA,f(x))
  return(p)
}

# Calculate log likelihood of discrete Weibull, conditional on being in the
# right/upper tail
# Input: Data vector, distributional parameters, lower cut-off
# Output: Log likelihood
discweib.loglike <- function(x,shape,scale,threshold=0) {
  sum(ddiscweib(x,shape,scale,threshold,log=TRUE))
}


# Fit discrete Weibull distributional, conditional on being in the right/upper
# tail, by numerically maximizing the likelihood
# Input: Data vector, lower threshold
# Output: List giving distribution type ("discweib"), estimate parameters,
#	  information on fit and data set
# Requires: weibull.R (to get starting point for optimization)
discweib.fit <- function(x,threshold=0) {
  # Start off with a rough-and-ready estimator for continuous data
  theta_0 <- weibull.est.shape.inefficient(x)
  # Trim the data set
  x <- x[x>=threshold]
  n <- length(x)
  # define the likelihood
  negloglike <- function(theta) {
    -discweib.loglike(x,theta[1],theta[2],threshold)
  }
  # optimize
  est <- nlm(f=negloglike,p=theta_0)
  fit <- list(type="discweib", shape=est$estimate[1], scale=est$estimate[2],
              loglike =-est$minimum, threshold=threshold,
              samples.over.threshold=n)
  return(fit)
}

