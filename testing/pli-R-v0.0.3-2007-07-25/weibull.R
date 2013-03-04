# Functions for estimation of a stretched exponential or Weibull distribution

# The use of a Weibull distribution to model values above a specified lower
# threshold can be done in one of two ways.  The first is simply to shift the
# the standard Weibull, i.e, to say that x-threshold ~ weibull.
# The other is to say that Pr(X|X>threshold) = weibull(X|X>threshold), i.e.,
# that the right tail follows the same functional form as the right tail of a
# Weibull, without necessarily having the same probability of being in the tail.
# These will be called the "shift" and "tail" methods respectively.
# The shift method is, computationally, infinitely easier, but not so suitable
# for our purposes.

# The basic R system provides dweibull (density), pweibull (cumulative
# distribution), qweibull (quantiles) and rweibull (random variate generation)

### Function for fitting:
# weibull.fit			Fit Weibull to data with choice of methods
### Distributional functions, per R standards:
# dweibull.tail			Tail-conditional probability density
# pweibull.tail			Tail-conditional cumulative distribution
### Backstage functions, not for users:
# weibull.fit.tail		Fit by maximizing tail-conditional likelihood
#				(default)
# weibull.fit.eqns		Fit by solving ML estimating equations for
#				shifted Weibull
# weibull.est.shape.inefficient Inefficient estimator of shape for shifted
#				Weibull, used to initialize fit.eqns
# weibull.est.scale.from.shape  MLE of scale given shape for shifted Weibull
# weibull.loglike.shift		Log-likelihood of shifted Weibull
# weibull.loglike.tail		Tail-conditional log-likelihood


# Fit Weibull to data
# Wrapper for functions implementing different ML methods
# Input: Data vector, lower threshold, method flag
# Output: List giving distribution type ("weibull"), parameters, log-likelihood
weibull.fit <- function(x, threshold = 0,method="tail") {
  x <- x[x>=threshold]
  switch(method,
    tail = {
      # Estimate parameters by direct maxmization of the tail-conditional
      # log-likelihood
      fit <- weibull.fit.tail(x,threshold)
    },
    eqns = {
      # Estimate parameters by solving the ML estimating equations of a shifted
      # Weibull
      fit <- weibull.fit.eqns(x,threshold)
    },
    {
      cat("Unknown method\n")
      fit <- NA})
  return(fit)
}



# Tail-conditional probability density function
# Returns NA on values < threshold
# Input: Data vector, distributional parameters, log flag
# Output: Vector of (log) probability densities
dweibull.tail <- function(x,shape,scale,threshold=0,log=FALSE) {
  c <- pweibull(threshold,shape=shape,scale=scale,lower.tail=FALSE,log.p=log)
  if (log) {
    f <- function(y) {dweibull(y,shape,scale,log=TRUE) - c}
  } else {
    f <- function(y) {dweibull(y,shape,scale)/c}
  }
  d <- ifelse(x<threshold,NA,f(x))
  return(d)
}

# Tail-conditional cumulative distribution function
# Returns NA on values < threshold
# Input: Data vector, distributional parameters, usual flags
# Output: Vector of (log) cumulative probabilities
pweibull.tail <- function(x,shape,scale,threshold=0,lower.tail=TRUE,
                          log.p=FALSE) {
  c <- pweibull(threshold,shape,scale,lower.tail=FALSE)
  c.log <- pweibull(threshold,shape,scale,lower.tail=FALSE,log.p=TRUE)
  if ((!lower.tail)&&(!log.p)) {
    f <- function(x) {pweibull(x,shape,scale,lower.tail=FALSE)/c}
  }
  if ((!lower.tail)&&(log.p)) {
    f <- function(x) {pweibull(x,shape,scale,lower.tail=FALSE,log.p=TRUE) - c.log}
  }
  if ((lower.tail)&&(!log.p)) {
    f <- function(x) {1 - pweibull(x,shape,scale,lower.tail=FALSE)/c}
  }
  if ((lower.tail)&&(log.p)) {
    f <- function(x) {log(1 - pweibull(x,shape,scale,lower.tail=FALSE)/c)}
  }
  p <- ifelse(x<threshold,NA,f(x))
  return(p)
}




# Fit Weibull to data by maximizing tail-conditional likelihood
  # CONSTRAINTS: The shape and scale parameters must both be positive
  # This will not necessarily give a _stretched_ exponential which would be
  # a shape < 1
# Input: Data vector, lower threshold
# Output: List giving distribution type ("weibull"), parameters, log-likelihood
weibull.fit.tail <- function(x,threshold=0) {
  # Use the whole data to produce initial estimates, via the estimating equation
  initial_fit <- weibull.fit.eqns(x)
  theta_0 <- c(initial_fit$shape, initial_fit$scale)
  # Apply constraints: if we're outside the feasible set, default to a
  # standardized exponential
  if (theta_0[1] < 0) { theta_0[1] = 1 }
  if (theta_0[2] < 0) { theta_0[2] = 1 }
  # Now threshold and directly minimize the negative log likelihood
  x <- x[x>= threshold]
  n <- length(x)
  negloglike <- function(theta) {
    -weibull.loglike.tail(x,threshold,shape=theta[1],scale=theta[2])
  }
  ui <- rbind(c(1,0),c(0,1))
  ci <- c(0,0)
  est <- constrOptim(theta=theta_0, f=negloglike, grad=NULL, ui=ui, ci=ci)
  fit <- list(type="weibull", shape=est$par[1], scale=est$par[2],
              loglike = -est$value, samples.over.threshold=n)
  return(fit)
}  

# Fit shifted Weibull to data by solving ML estimating equations
# Input: Data vector, lower threshold
# Output: List giving distribution type ("weibull"), parameters, log-likelihood
weibull.fit.eqns <- function(x, threshold = 0) {
  # ML estimating equations of the shape and scale of the Weibull distribution,
  # taken from Johnson and Kotz, ch. 20
  # This assumes that x-threshold is Weibull-distributed
  x <- x[x>=threshold]
  x <- x-threshold
  x.log <- log(x) # This will be needed repeatedly
  n<-length(x) # So will this
  h <- sum(x.log)/n # And this
  scale_from_shape <- function(shape) {weibull.est.scale.from.shape(x,shape)}
  # Note that the estimation of the shape parameter is only implicit,
  # through a transcendental equation.
  initial_estimates <- weibull.est.shape.inefficient(x)
  shape <- initial_estimates[1]
  scale <- initial_estimates[2]
  map <- function(c) {(sum((x^c) * x.log)/sum(x^c) - h)^(-1)}
  estimating_equation <- function(c) { (c - map(c))^2 }
  shape <- nlm(f=estimating_equation,p=shape)$estimate
  scale <- scale_from_shape(shape)
  loglike <- weibull.loglike.shift(x,threshold,shape,scale)
  fit <- list(type="weibull", shape=shape, scale=scale, loglike=loglike, samples.over.threshold=n)
  return(fit)
}

# Log-likelihood of shifted Weibull distribution
# Input: Data vector, parameters, lower threshold
# Output: real-valued log-likelihood
weibull.loglike.shift <- function(x, shape, scale, threshold = 0) {
  # Assumes x - threshold is Weibull-distributed
  # See Johnson and Kotz for more; they call the lower threshold xi_0
  x <- x[x>=threshold]
  x <- x-threshold
  L <- sum(dweibull(x,shape,scale,log=TRUE))
  return(L)
}

# Tail-conditional log-likelihood
# Input: Data vector, parameters, lower threshold
# Output: Real-valued log-likelihood
weibull.loglike.tail <- function(x, shape, scale, threshold = 0) {
  # We want p(X=x|X>=threshold) = p(X=x)/Pr(X>=threshold)
  x <- x[x>=threshold]
  n <- length(x)
  Like <- weibull.loglike.shift(x,shape,scale)
  ThresholdProb <- pweibull(threshold,shape,scale,log=TRUE,lower.tail = FALSE)
  L <- Like - n*ThresholdProb
  return(L)
}

# Inefficient estimator of the shape parameter of a Weibull, plus scale
# Based on the moments of log of the data
# Input: Data vector, lower threshold
# Output: Real-valued estimates of shape and scale
weibull.est.shape.inefficient <- function(x,threshold=0) {
  # The follow is not an efficient estimator of the shape, but serves to start
  # the approximation process off.  (See Johnson and Katz, ch. 20, section 6.3,
  # "Estimators Based on Distribution of log X".)
  x <- x-threshold
  shape <- (sqrt(6)/pi)*sd(log(x))
  scale <- weibull.est.scale.from.shape(x,shape)
  c(shape,scale)
}

# Maximum likelihood estimate of a Weibull scale parameter, given shape
# Input: Data vector, shape parameter
# Output: Real-valued scale parameter
weibull.est.scale.from.shape <- function(x,shape) {
  # Given a value of the shape parameter, return the corresponding
  # MLE of the scale parameter
  n <- length(x)
  scale <- ((1/n)*sum(x^shape))^(1/shape)
  return(scale)
}
