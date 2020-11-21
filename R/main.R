#' @useDynLib gfconfExtremes
#' @importFrom Rcpp evalCpp
NULL


#' Title
#'
#' @param X numeric vector of data
#' @param beta vector of probabilities corresponding to the quantiles to be 
#'   estimated
#' @param i index for the initial threshold at the \code{X(i)} order statistic
#' @param gamma.init starting value for gamma in the MCMC
#' @param sigma.init starting value for sigma in the MCMC
#' @param sd.gamma standard deviation for the proposed gamma
#' @param sd.sigma standard deviation for the proposed sigma
#' @param p1 probability that the MCMC will propose a new \code{(gamma,sigma)}; 
#'   \code{(1-p1)} would be the probability that the MCMC chain will propose a 
#'   new index for a new threshold
#' @param p2 probability that the new index proposed will be larger than the 
#'   current index
#' @param lambda1 the small jump the index variable will make
#' @param lambda2 the large jump the index variable will make; happens 1 of 
#'   every 10 iterations
#' @param Jnumb number of subsamples that are taken from the Jacobian
#' @param iter number of iterations per chain (burnin excluded)
#' @param burnin number of the first MCMC iterations discarded
#' @param thin thinning number for the MCMC chain. (e.g. if it is 1 no iteration 
#' is skipped)
#'
#' @return
#' @export
#'
#' @examples xx
gfigpd <- function(
  X, beta, i = floor(0.85 * length(X)), 
  gamma.init = NA, sigma.init = NA, sd.gamma = NA, sd.sigma = NA, 
  p1 = 0.9, p2 = 0.5, lambda1 = 2, lambda2 = 10, Jnumb = 50L, 
  iter = 10000L, burnin = 2000L, thin = 6L) {
  
  stopifnot(thin >= 1L)

  X <- sort(X) # -->> so there's no need to sort in C++
  n <- length(X)
  
  # Intialize the default values for the tuning parameters of the MCMC chain.
  if(is.na(gamma.init) || is.na(sigma.init)) {
    mle.fit <- gpd.fit(X, X[i], show = FALSE)
    if (is.na(gamma.init)) gamma.init <- mle.fit$mle[2L]
    if (is.na(sigma.init)) sigma.init <- mle.fit$mle[1L]
  }
  if(is.na(sd.gamma)) sd.gamma <- 2 * abs(g) / 3
  if(is.na(sd.sigma)) sd.sigma <- 2 * s / 3
  
  skip.number <- thin - 1L
  number.iterations <- (skip.number + 1L) * iter + burnin
  
  # run the MCMC chain.
  x.t <- gfconfExtremes:::MCMCchain(
    X, beta, gamma.init, sigma.init, i, 
    p1, p2, lambda1, lambda2, sd.gamma, sd.sigma,
    number.iterations, burnin, Jnumb, 666L
  )
  
  # Thin the chain by keeping every skip.number+1 iteration of the MCMC chain
  number.iterations <- nrow(x.t)
  every.ith <- c(TRUE, rep(FALSE, skip.number))
  eliminate.vector <-
    rep(
      every.ith,
      ceiling((number.iterations) / length(every.ith))
    )[1:number.iterations]
  
  x.t <- x.t[eliminate.vector, ]
  

  # Indictor for the acceptance rate. // pas besoin avec coda::rejectionRate
  ii <- integer(nrow(x.t))
  for (i in 1L:(nrow(x.t) - 1L)) {
    if (x.t[i, 4L] != x.t[i + 1L, 4L]) {
      ii[i + 1L] <- 1L
    }
  }
  acceptance.rate <- mean(ii)
  cat("acceptance rate: ", acceptance.rate)
  
  x.t[, 4L:(3L + length(beta))] <- x.t[, 4L:(3L + length(beta))] + X[1L]
  
  
  
  return(x.t)
}
