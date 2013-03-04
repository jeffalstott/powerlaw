# Functions for fitting discrete exponential distributions to (right) tails
# of data

### Function for fitting:
# discexp.fit		Fit discrete exponential by maximum likelihood
### Function for distribution, per R standard:
# ddiscexp		Probability mass function
### Backstage function, not intended for user:
# discexp.loglike	Calculate log likelihood

# Probability mass function for discrete exponential distribution, conditional
# on being in the tail
# Input: Data vector, distributional parameters, log flag
# Output: Vector of (log) probabilities
ddiscexp <- function(x, lambda, threshold=0, log=FALSE) {
  if (log) {
    C <- log(1-exp(-lambda)) + lambda*threshold
    f <- function(x) {C -lambda*x}
  } else {
    C <- (1-exp(-lambda))*exp(lambda*threshold)
    f <- function(x) {C*exp(-lambda*x)}
  }
  d <- ifelse(x<threshold,NA,f(x))
  return(d)
}

# Log likelihood of tail-conditional discrete exponential
# Input: Data vector, distributional parameters
# Output: Log likelihood
discexp.loglike <- function(x, lambda, threshold=0) {
  return(sum(ddiscexp(x,lambda,threshold,log=TRUE)))
}

# Fit discrete exponential to tail of data via numerical likelihood
# maximization
# Input: Data vector, lower cut-off
# Output: List giving fitted parameter values and information on data and
#         fitting process
discexp.fit <- function(x,threshold=0) {
  x <- x[x>=threshold]
  n <- length(x)
  lambda <- log(1+n/sum(x-threshold))	# Moment-based estimate to start the
					# optimization off
  loglike <- discexp.loglike(x,lambda,threshold)
  fit <- list(type="discexp", lambda=lambda, loglike=loglike,
              threshold=threshold, method="formula", samples.over.threshold=n)
  return(fit)
}
