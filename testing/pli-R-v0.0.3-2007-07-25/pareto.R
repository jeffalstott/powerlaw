#### Functions for continuous power law or Pareto distributions
# Revision history at end of file

### Standard R-type functions for distributions:
# dpareto		Probability density
# ppareto		Probability distribution (CDF)
# qpareto		Quantile function
# rpareto		Random variable generation
### Functions for fitting:
# pareto.fit			Fit Pareto to data
# pareto.fit.ml			Fit Pareto to data by maximum likelihood
#                               --- not for direct use, call pareto.fit instead
# pareto.loglike		Calculate log-likelihood under Pareto
# pareto.fit.regression.cdf	Fit Pareto data by linear regression on
#				log-log CDF (disrecommended)
#                               --- not for direct use, call pareto.fit instead
# loglogslope			Fit Pareto via regression, extract scaling
#				exponent
# loglogrsq			Fit Pareto via regression, extract R^2
### Functions for visualization:
# plot.eucdf.loglog		Log-log plot of the empirical upper cumulative
#				distribution function, AKA survival function
# plot.survival.loglog		Alias for plot.eucdf.loglog
### Back-stage functions, not intended for users:
# unique_values			Find the indices representing unique values
#				in a sorted list (used in regression fit)

# Probability density of Pareto distributions
# Gives NA on values below the threshold
# Input: Data vector, lower threshold, scaling exponent, "log" flag
# Output: Vector of (log) probability densities
dpareto <- function(x, threshold = 1, exponent, log=FALSE) {
  # Avoid doing limited-precision arithmetic followed by logs if we want
  # the log!
  if (!log) {
    prefactor <- (exponent-1)/threshold
    f <- function(x) {prefactor*(x/threshold)^(-exponent)}
  } else {
    prefactor.log <- log(exponent-1) - log(threshold)
    f <- function(x) {prefactor.log -exponent*(log(x) - log(threshold))}
  }
  d <- ifelse(x<threshold,NA,f(x))
  return(d)
}

# Cumulative distribution function of the Pareto distributions
# Gives NA on values < threshold
# Input: Data vector, lower threshold, scaling exponent, usual flags
# Output: Vector of (log) probabilities
ppareto <- function(x, threshold=1, exponent, lower.tail=TRUE, log.p=FALSE) {
  if ((!lower.tail) && (!log.p)) {
    f <- function(x) {(x/threshold)^(1-exponent)}
  }
  if ((lower.tail) && (!log.p)) {
    f <- function(x) { 1 - (x/threshold)^(1-exponent)}
  }
  if ((!lower.tail) && (log.p)) {
    f <- function(x) {(1-exponent)*(log(x) - log(threshold))}
  }
  if ((lower.tail) && (log.p)) {
    f <- function(x) {log(1 - (x/threshold)^(1-exponent))}
  }
  p <- ifelse(x < threshold, NA, f(x))
  return(p)
}

# Quantiles of Pareto distributions
# Input: vector of probabilities, lower threshold, scaling exponent, usual flags
# Output: Vector of quantile values
qpareto <- function(p, threshold=1, exponent, lower.tail=TRUE, log.p=FALSE) {
  # Quantile function for Pareto distribution
  # P(x) = 1 - (x/xmin)^(1-a)
  # 1-p = (x(p)/xmin)^(1-a)
  # (1-p)^(1/(1-a)) = x(p)/xmin
  # xmin*((1-p)^(1/(1-a))) = x(p)
  # Upper quantile:
  # U(x) = (x/xmin)^(1-a)
  # u^(1/(1-a)) = x/xmin
  # xmin * u^(1/(1-a)) = x
  # log(xmin) + (1/(1-a)) log(u) = log(x)
  if (log.p) {
    p <- exp(p)
  }
  if (lower.tail) {
    p <- 1-p
  }
  # This works, via the recycling rule
  # q<-(p^(1/(1-exponent)))*threshold
  q.log <- log(threshold) + (1/(1-exponent))*log(p)
  q <- exp(q.log)
  return(q)
}

# Generate Pareto-distributed random variates
# Input: Integer size, lower threshold, scaling exponent
# Output: Vector of real-valued random variates
rpareto <- function(n, threshold=1, exponent) {
  # Using the transformation method, because we know the quantile function
  # analytically
  # Consider replacing with a non-R implementation of transformation method
  ru <- runif(n)
  r<-qpareto(ru,threshold,exponent)
  return(r)
}

# Estimate scaling exponent of Pareto distribution
# A wrapper for functions implementing actual methods
# Input: data vector, lower threshold, method (likelihood or regression,
#        defaulting to former)
# Output: List indicating type of distribution ("exponent"), parameters,
#         information about fit (depending on method), OR a warning and NA
#         if method is not recognized
pareto.fit <- function(data, threshold, method="ml") {
  switch(method,
    ml = { return(pareto.fit.ml(data,threshold)) },
    regression.cdf = { return(pareto.fit.regression.cdf(data,threshold)) },
    { cat("Unknown method\n"); return(NA)}
  )
}

# Estimate scaling exponent of Pareto distribution by maximum likelihood
# Input: Data vector, lower threshold
# Output: List giving distribution type ("pareto"), parameters, log-likelihood
pareto.fit.ml <- function (data, threshold) {
  data <- data[data>=threshold]
  n <- length(data)
  x <- data/threshold
  alpha <- 1 + n/sum(log(x))
  loglike = pareto.loglike(data,threshold,alpha)
  fit <- list(type="pareto", exponent=alpha, xmin=threshold, loglike = loglike)
  return(fit)
}

# Calculate log-likelihood under a Pareto distribution
# Input: Data vector, lower threshold, scaling exponent
# Output: Real-valued log-likelihood
pareto.loglike <- function(x, threshold, exponent) {
  L <- sum(dpareto(x, threshold = threshold, exponent = exponent, log = TRUE))
  return(L)
}

# Log-log plot of the survival function (empirical upper CDF) of a data set
# Input: Data vector, lower limit, upper limit, graphics parameters
# Output: None (returns NULL invisibly)
plot.survival.loglog <- function(x,from=min(x),to=max(x),...) {
	plot.eucdf.loglog(x,from,to,...)
}
plot.eucdf.loglog <- function(x,from=min(x),to=max(x),...) {
	# Exploit built-in R function to get ordinary (lower) ECDF, Pr(X<=x)
	x.ecdf <- ecdf(x)
	# Now we want Pr(X>=x) = (1-Pr(X<=x)) + Pr(X==x)
        # If x is one of the "knots" of the step function, i.e., a point with
	# positive probability mass, should add that in to get Pr(X>=x)
	# rather than Pr(X>x)
	away.from.knot <- function(y) { 1 - x.ecdf(y) }
	at.knot.prob.jump <- function(y) {
		x.knots = knots(x.ecdf)
		# Either get the knot number, or give zero if this was called
		# away from a knot
		k <- match(y,x.knots,nomatch=0)
		if ((k==0) || (k==1)) { # Handle special cases
			if (k==0) {
				prob.jump = 0 # Not really a knot
			} else {
				prob.jump = x.ecdf(y) # Special handling of first knot
			}
		} else {
			prob.jump = x.ecdf(y) - x.ecdf(x.knots[(k-1)]) # General case
		}
		return(prob.jump)
	}
	# Use one function or the other
	x.eucdf <- function(y) {
		baseline = away.from.knot(y)
		jumps = sapply(y,at.knot.prob.jump)
		ifelse (y %in% knots(x.ecdf), baseline+jumps, baseline)
	}
	plot(x.eucdf,from=from,to=to,log="xy",...)
	invisible(NULL)
}


### The crappy linear regression way to fit a power law
# The common procedure is to fit to the binned density function, which is even
# crappier than to fit to the complementary distribution function; this
# currently only implements the latter

# First, produce the empirical complementary distribution function, as
# a pair of lists, {x}, {C(x)}
# Then regress log(C) ~ log(x)
# and report the slope and the R^2
# Input: Data vector, threshold
# Output: List with distributional parameters and information about the
#         fit
pareto.fit.regression.cdf <- function(x,threshold=1) {
  x <- x[x>=threshold]
  n <- length(x)
  x <- sort(x)
  uniqs <- unique_values(x)
  distinct_x <- x[uniqs]
  upper_probs <- ((n+1-uniqs))/n
  log_distinct_x <- log(distinct_x)
  log_upper_probs <- log(upper_probs)
  # so if one unique index was n, this would give prob 1/n there, and if one
  # unique index is 1 (which it should always be!), it will give prob 1 there
  loglogfit <- lm(log_upper_probs ~ log_distinct_x)
  intercept <- coef(loglogfit)[1] # primarily useful for plotting purposes
  slope <- -coef(loglogfit)[2] # Remember sign of parameterization
  # But that's the exponent of the CDF, that of the pdf is one larger
  # and is what we're parameterizing by
  slope <- slope+1
  r2 <- summary(loglogfit)$r.squared
  loglike <- pareto.loglike(x, threshold, slope)
  result <- list(type="pareto", exponent = slope, rsquare = r2,
                 log_x = log_distinct_x, log_p = log_upper_probs,
                 intercept = intercept, loglike = loglike, xmin=threshold)
  return(result)
}

# Wrapper function to just get the exponent estimate
loglogslope <- function(x,threshold=1) {
  llf <- pareto.fit.regression.cdf(x,threshold)
  exponent <- llf$exponent
  return(exponent)
}

# Wrapper function to just get the R^2 values
loglogrsq <- function(x,threshold=1) {
  llf <- pareto.fit.regression.cdf(x,threshold)
  r2 <- llf$rsquare
  return(r2)
}

# Function to take a sorted list of values, and return only the indices
# to unique values
# Called in finding the empirical complementary distribution function
# If a value is unique, return its index
# If a value is repeated, return its lowest index --- this is intended
# for working with the empirical complementary distribution function
# Input: a SORTED list of (real) numbers
# Output: a list of the indices of the input which mark distinct values
unique_values <- function(a_sorted_list) {
    # See which members of the list are strictly less than their successor
    n <- length(a_sorted_list)
    is_lesser <- a_sorted_list[2:n] > a_sorted_list[1:(n-1)]
    # convert to index numbers
    most_indices <- 1:(n-1)
    my_indices <- most_indices[is_lesser]
    # Remember that we've checked a list shortened by one from the start
    my_indices <- my_indices+1
    # Remember that the first item in the list has to be included
    my_indices <- c(1,my_indices)
    return(my_indices)
}


# Revision history:
# no release	2003		First draft
# v 0.0		2007-06-04	First release
# v 0.0.1	2007-06-29	Fixed "not" for "knot" typo, thanks to
#				Nicholas A. Povak for bug report
# v 0.0.2	2007-07-22	Fixed bugs in plot.survival.loglog, thanks to
#						Stefan Wehrli for the report
