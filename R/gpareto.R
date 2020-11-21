#' Generalized pareto distribution
#' @description Density, distribution function, quantile function, and random 
#'   generation for the generalized pareto distribution.
#'   
#' @param u numeric vector
#' @param q numeric vector of quantiles
#' @param p numeric vector of probabilities
#' @param n positive integer, the desired number of simulations
#' @param c,d,kappa,tau parameters; they must be strictly positive numbers, 
#'   except \code{kappa} which can take any value
#' @param log logical, whether to return the log-density
#' @param method the method of random generation, \code{"mixture"} or 
#'   \code{"arou"}; only a positive \code{kappa} is allowed for the 
#'   \code{"mixture"} method, but this method is faster
#'   
#' @references 
#' \itemize{
#'   \item Marwa Hamza & Pierre Vallois. 
#'     \emph{On Kummer’s distributions of type two and generalized pareto 
#'           distributions}.
#'     Statistics & Probability Letters 118 (2016), pp. 60-69.
#'     <doi:10.1016/j.spl.2016.03.014>
#'   \item James J. Chen & Melvin R. Novick.
#'     \emph{Bayesian Analysis for Binomial Models with Generalized pareto Prior 
#'           Distributions}.
#'     Journal of Educational Statistics 9, No. 2 (1984), pp. 163-175.
#'     <doi:10.3102/10769986009002163>
#' }
#' 
#' @examples library(gpareto)
#' curve(dgpareto(x, 4, 12, 10, 0.01), axes = FALSE, lwd = 2)
#' axis(1)
#' 
#' @importFrom stats qpareto
#' @importFrom Runuran uq
#' 
#' @rdname GPareto
#' @name GPareto
#' @export
dgpareto <- function(x, mu, gamma, sigma, log = FALSE){ 
  stopifnot(gamma >= 0, sigma > 0)
  out <- numeric(length(x))
  less_than_mu <- q < mu
  in_support <- !less_than_mu
  if(any(in_support)){
    x <- x[in_support]
    z <- (x - mu) / sigma
    out[in_support] <- if(gamma == 0){
      if(log) -z else exp(-z)
    }else{
      if(log){
        (-1/gamma-1) * log1p(gamma*z) - log(sigma)
      }else{
        (1 + gamma*z)^(-1/gamma-1) / sigma
      }
    }
  }
  out
}

#' @rdname GPareto
#' @export
pgpareto <- function(q, mu, gamma, sigma){ 
  stopifnot(gamma >= 0, sigma > 0)
  out <- numeric(length(q))
  less_than_mu <- q < mu
  in_support <- !less_than_mu
  if(any(in_support)){
    q <- q[in_support]
    z <- (q - mu) / sigma
    out[in_support] <- if(gamma == 0){
      1 - exp(-z)
    }else{
      1 - (1 + gamma*z)^(-1/gamma)
    }
  }
  out
}

#' @rdname GPareto
#' @export
rgpareto <- function(n, mu, gamma, sigma){ 
  qgpareto(runif(n), mu, gamma, sigma)
}

#' @rdname GPareto
#' @export
qgpareto <- function(p, mu, gamma, sigma){
  stopifnot(all(p >= 0 & p <= 1))
  stopifnot(gamma >= 0, sigma > 0)
  if(gamma == 0){
    mu - sigma * log1p(-p)
  }else{
    mu + sigma * ((1-p)^(-gamma) - 1) / gamma
  }
}
